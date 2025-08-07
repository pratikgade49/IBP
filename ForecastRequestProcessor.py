import logging
import requests
import json
import urllib3
from typing import List
from Algorithms import calculate_forecast
import os
from configparser import ConfigParser
import datetime
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('forecast_processor.log')
    ]
)
logger = logging.getLogger(__name__)

urllib3.disable_warnings()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Read config file
try:
    config_object = ConfigParser()
    config_object.read("server.cfg")
    
    # User credentials and server url
    SERVER_URL = config_object["SERVICECONFIG"]["server_url"]
    USERNAME = config_object["AUTHCONFIG"]["username"]
    PASSWORD = config_object["AUTHCONFIG"]["password"]
    
    TASK_URL = f"https://{SERVER_URL}/sap/opu/odata4/ibp/api_dmdfcstextalg/srvd_a2x/ibp/api_dmdfcstextalg/0001/Request"
    RESULT_URL = f"https://{SERVER_URL}/sap/opu/odata4/ibp/api_dmdfcstextalg/srvd_a2x/ibp/api_dmdfcstextalg/0001/Result"
    DATA_URL = f"https://{SERVER_URL}/sap/opu/odata4/ibp/api_dmdfcstextalg/srvd_a2x/ibp/api_dmdfcstextalg/0001/Input"
    
    logger.info("Successfully loaded configuration")
except Exception as e:
    logger.error(f"Failed to load configuration: {str(e)}", exc_info=True)
    raise

def create_result_string(results: List) -> str:
    """Concatenates the results with ';' to a string."""
    logger.debug(f"Creating result string from list of length {len(results)}")
    try:
        # Ensure all values are converted to strings
        cleaned_results = []
        for item in results:
            if isinstance(item, float) and not np.isfinite(item):
                cleaned_results.append("NULL")
                logger.warning("Replaced non-finite value with NULL in final output")
            else:
                cleaned_results.append(str(item))
                
        result = ";".join(cleaned_results)
        logger.debug(f"Created result string with length {len(result)}")
        return result
    except Exception as e:
        logger.error(f"Error in create_result_string: {str(e)}", exc_info=True)
        return ";".join(["NULL"] * len(results))

def get_remaining_data(planning_objects: List, url: str, cookies: object) -> List:
    """Requests the next datachunk until all data arrived"""
    logger.debug(f"Getting remaining data from URL: {url}")
    try:
        data_get = requests.get(
            url, headers={"accept": "application/json"}, cookies=cookies, verify=False)

        if data_get.status_code == 200:
            if data_get.json()["value"]:
                planning_objects.extend(data_get.json()["value"])
                logger.debug(f"Added {len(data_get.json()['value'])} planning objects")
            else:
                logger.debug("No more data in response")
                return planning_objects

            if "@odata.nextLink" in data_get.json().keys():
                logger.debug("Found nextLink, recursively fetching more data")
                return get_remaining_data(planning_objects,
                                          f"https://{SERVER_URL}{data_get.json()['@odata.nextLink']}", cookies)
            else:
                logger.debug("No more nextLink, returning complete data")
                return planning_objects
        else:
            logger.error(f"Failed to retrieve forecast data! Status code: {data_get.status_code}.")
            raise ConnectionError(f"HTTP {data_get.status_code}: {data_get.text}")
            
    except Exception as e:
        logger.error(f"Error in get_remaining_data: {str(e)}", exc_info=True)
        raise

def process_forecast_request(request_id: int):
    """Runs the forecasting process on the request."""
    logger.info(f"Starting process_forecast_request for request ID: {request_id}")
    try:
        logger.debug(f"Making initial request to TASK_URL for request {request_id}")
        request_get = requests.get(f"{TASK_URL}?$filter=RequestID%20eq%20{request_id}&$expand=_TimePeriod",
                                   headers={"accept": "application/json", "x-sap-security-session": "create"}, 
                                   auth=(USERNAME, PASSWORD), verify=False)

        if request_get.status_code == 200:
            logger.info("Successfully retrieved forecast request details")
            request_data = request_get.json()["value"][0]
            cookies = request_get.cookies

            algorithm_name = request_data["AlgorithmName"]
            time_period_data = request_data["_TimePeriod"]
            logger.info(f"Processing algorithm: {algorithm_name}")

            # Datetime list [(start_datetime, end_datetime)]
            date_list = [(datetime.datetime.strptime(time_period["StartTimeStamp"], "%Y-%m-%dT%H:%M:%SZ"),
                          datetime.datetime.strptime(time_period["EndTimeStamp"], "%Y-%m-%dT%H:%M:%SZ"))
                          for time_period in time_period_data]
            logger.debug(f"Processed {len(date_list)} time periods")

            parameters = {}
            if request_data["AlgorithmParameter"]:
                param_list = []
                for parameter in request_data["AlgorithmParameter"].split(";"):
                    if "=" in parameter:
                        parts = list(map(str.strip, parameter.split('=', 1)))
                        if len(parts) == 2:
                            param_list.append(parts)
                parameters = dict(param_list)
                logger.debug(f"Processed algorithm parameters: {parameters}")

            historical_periods = request_data["HistoricalPeriods"]
            forecast_periods = request_data["ForecastPeriods"]
            logger.info(f"Historical periods: {historical_periods}, Forecast periods: {forecast_periods}")

            logger.debug(f"Making data request to DATA_URL for request {request_id}")
            data_get = requests.get(f"{DATA_URL}?$filter=RequestID%20eq%20{request_id}&$expand=_AlgorithmDataInput,_MasterData",
                                    headers={"accept": "application/json"}, cookies=cookies, verify=False)

            results = {}
            if data_get.status_code == 200:
                logger.info("Successfully retrieved initial forecast data")
                planning_objects = data_get.json()["value"]
                logger.debug(f"Initial planning objects count: {len(planning_objects)}")

                if "@odata.nextLink" in data_get.json().keys():
                    try:
                        logger.debug("Fetching additional planning objects via nextLink")
                        planning_objects = get_remaining_data(
                            planning_objects, f"https://{SERVER_URL}{data_get.json()['@odata.nextLink']}", cookies)
                        logger.info(f"Total planning objects after additional fetches: {len(planning_objects)}")
                    except ConnectionError:
                        logger.error("Failed to get remaining data")
                        return

                # Output format
                output = {
                    "RequestID": request_id,
                    "_AlgorithmDataOutput": [],
                    "_Message": [],
                }

                for planning_object in planning_objects:
                    try:
                        group_id = planning_object["GroupID"]
                        logger.debug(f"Processing planning object GroupID: {group_id}")
                        results = calculate_forecast(
                            planning_object, algorithm_name, parameters,
                            historical_periods, forecast_periods, date_list)
                        
                        if not results:
                            logger.error(f"No results returned for GroupID {group_id}")
                            continue
                            
                        # Output Key Figures
                        for key_figure_name, key_figure_result in results.items():
                            # Final validation
                            if any(isinstance(v, float) and not np.isfinite(v) for v in key_figure_result):
                                logger.error(f"Non-finite values found in {key_figure_name} for GroupID {group_id}")
                                key_figure_result = ["NULL" if isinstance(v, float) and not np.isfinite(v) else v 
                                                   for v in key_figure_result]
                                
                            key_figure_data = {
                                "RequestID": request_id,
                                "GroupID": group_id,
                                "SemanticKeyFigure": key_figure_name,
                                "ResultData": create_result_string(key_figure_result),
                            }
                            output["_AlgorithmDataOutput"].append(key_figure_data)

                        # Success message
                        message = {
                            "RequestID": request_id,
                            "GroupID": group_id,
                            "MessageSequence": 1,
                            "MessageType": "I",
                            "MessageText": "Forecast calculated successfully",
                        }
                        output["_Message"].append(message)
                        logger.debug(f"Added results for GroupID: {group_id}")
                    except Exception as e:
                        logger.error(f"Error processing planning object GroupID {group_id}: {str(e)}", exc_info=True)
                        # Add error message
                        error_msg = {
                            "RequestID": request_id,
                            "GroupID": group_id,
                            "MessageSequence": 1,
                            "MessageType": "E",
                            "MessageText": f"Forecast calculation failed: {str(e)}",
                        }
                        output["_Message"].append(error_msg)
                        continue

                # Header message
                if output["_AlgorithmDataOutput"]:
                    msg_header = {
                        "RequestID": request_id,
                        "GroupID": -1,
                        "MessageSequence": 1,
                        "MessageType": "I",
                        "MessageText": f"{algorithm_name} forecast completed for {len(planning_objects)} objects",
                    }
                    output["_Message"].append(msg_header)
                    logger.info("Successfully processed all planning objects")
                else:
                    msg_header = {
                        "RequestID": request_id,
                        "GroupID": -1,
                        "MessageSequence": 1,
                        "MessageType": "E",
                        "MessageText": f"Forecast calculation failed for all objects! Algorithm: {algorithm_name}",
                    }
                    output["_Message"].append(msg_header)
                    logger.error("Failed to generate any results for planning objects")

                output_json = json.dumps(output)
                logger.debug("Prepared output JSON for sending")

                logger.debug("Fetching CSRF token for result submission")
                token_request = requests.get(RESULT_URL, headers={"x-csrf-token": "fetch", "accept": "application/json"},
                                             cookies=cookies, verify=False)

                if token_request.status_code == 200:
                    logger.debug("Successfully retrieved CSRF token")
                    csrf_token = token_request.headers["x-csrf-token"]
                    result_send_post = requests.post(RESULT_URL, output_json, cookies=cookies,
                                                     headers={
                                                         "x-csrf-token": csrf_token, 
                                                         "x-sap-security-session": "delete",
                                                         "Content-Type": "application/json", 
                                                         "OData-Version": "4.0"
                                                     }, 
                                                     verify=False)

                    if result_send_post.status_code == 201:
                        logger.info(f"Forecast result for id {request_id} sent successfully! Status code: {result_send_post.status_code}.")
                    else:
                        logger.error(f"Failed to send forecast result! Status: {result_send_post.status_code}, Response: {result_send_post.text}")
                else:
                    logger.error(f"Failed to retrieve x-csrf token! Status code: {token_request.status_code}.")
            else:
                logger.error(f"Failed to retrieve forecast data! Status code: {data_get.status_code}.")
        else:
            logger.error(f"Failed to retrieve forecast model details! Status code: {request_get.status_code}")
            
    except Exception as e:
        logger.error(f"Error in process_forecast_request for request {request_id}: {str(e)}", exc_info=True)
    finally:
        logger.info(f"Completed processing for request ID: {request_id}")