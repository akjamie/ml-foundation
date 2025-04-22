# Machine learning - python basis

- Matplotlib
- Numpy
- Pandas


```
import time
import json
import inspect
import logging
from typing import Dict, List, Optional, Any, Union, Callable, Type, TypeVar
from pydantic import BaseModel, Field, create_model
from enum import Enum

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool, BaseTool
from langchain.tools.base import ToolException

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Base Response Schema Classes ---
class ApiResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"

class BaseApiResponse(BaseModel):
    """Base model for all API responses"""
    status: ApiResponseStatus = Field(..., description="Response status (success, error, partial)")
    request_id: Optional[str] = Field(None, description="Unique identifier for tracking the request")

class ErrorDetail(BaseModel):
    """Detailed error information"""
    code: str = Field(..., description="Error code identifying the error type")
    message: str = Field(..., description="Human-readable error message")
    field: Optional[str] = Field(None, description="Field that caused the error, if applicable")
    details: Optional[Any] = Field(None, description="Additional error details")

class ApiErrorResponse(BaseApiResponse):
    """Standard error response format"""
    status: ApiResponseStatus = Field(ApiResponseStatus.ERROR, const=True)
    errors: List[ErrorDetail] = Field(..., description="List of error details")
    
class ApiSuccessResponse(BaseApiResponse):
    """Standard success response format"""
    status: ApiResponseStatus = Field(ApiResponseStatus.SUCCESS, const=True)

# --- Parameter Validation ---
class ParameterValidator:
    """Validates parameters before API calls"""
    
    @staticmethod
    def validate(params: Dict[str, Any], schema: Type[BaseModel]) -> Dict[str, Any]:
        """Validate parameters against a Pydantic schema"""
        if params is None:
            params = {}
            
        # If params is already the correct type, use it directly
        if isinstance(params, schema):
            return params
            
        try:
            # Validate against schema and return
            validated = schema(**params)
            return validated
        except Exception as e:
            # Convert Pydantic validation errors to a more readable format
            error_messages = []
            if hasattr(e, 'errors'):
                for error in e.errors():
                    path = '.'.join(str(p) for p in error['loc'])
                    error_messages.append(f"{path}: {error['msg']}")
            
            if error_messages:
                raise ValueError(f"Parameter validation failed: {'; '.join(error_messages)}")
            else:
                raise ValueError(f"Parameter validation failed: {str(e)}")

# --- API Service Base Class ---
class ApiService:
    """Base class for API services with multiple endpoints"""
    
    def __init__(self, service_name: str, base_url: Optional[str] = None, 
                 auth_config: Optional[Dict[str, Any]] = None,
                 default_retry_config: Optional[Dict[str, Any]] = None):
        self.service_name = service_name
        self.base_url = base_url
        self.auth_config = auth_config or {}
        self.default_retry_config = default_retry_config or {
            "max_retries": 3,
            "retry_delay": 1,  # seconds
            "retry_backoff": 2  # exponential backoff multiplier
        }
        self._endpoints = {}
        
        # Auto-register endpoints based on methods with @endpoint decorator
        self._register_endpoints()
    
    def _register_endpoints(self):
        """Auto-register all methods decorated with @endpoint"""
        for name, method in inspect.getmembers(self, inspect.ismethod):
            if hasattr(method, '_endpoint_info'):
                endpoint_info = getattr(method, '_endpoint_info')
                self._endpoints[endpoint_info['name']] = {
                    'method': method,
                    'path': endpoint_info.get('path'),
                    'description': endpoint_info.get('description') or method.__doc__,
                    'param_schema': endpoint_info.get('param_schema'),
                    'response_schema': endpoint_info.get('response_schema'),
                    'retry_config': endpoint_info.get('retry_config') or self.default_retry_config
                }
    
    def get_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        """Get information about a specific endpoint"""
        if endpoint_name not in self._endpoints:
            raise ValueError(f"Endpoint '{endpoint_name}' not found in service '{self.service_name}'")
        return self._endpoints[endpoint_name]
    
    def get_endpoint_names(self) -> List[str]:
        """Get all available endpoint names"""
        return list(self._endpoints.keys())
    
    def call_endpoint(self, endpoint_name: str, **params) -> Dict[str, Any]:
        """Call a specific endpoint with parameters"""
        if endpoint_name not in self._endpoints:
            raise ValueError(f"Endpoint '{endpoint_name}' not found in service '{self.service_name}'")
            
        endpoint = self._endpoints[endpoint_name]
        
        # Validate parameters if a schema is provided
        if endpoint['param_schema']:
            validated_params = ParameterValidator.validate(params, endpoint['param_schema'])
            # Convert Pydantic model to dict if needed
            if isinstance(validated_params, BaseModel):
                params = validated_params.dict()
            else:
                params = validated_params
        
        # Call the endpoint method
        method = endpoint['method']
        return method(**params)
    
    def list_endpoints(self) -> Dict[str, str]:
        """List all endpoints with descriptions"""
        return {name: info['description'] for name, info in self._endpoints.items()}

# --- Endpoint Decorator ---
def endpoint(name: str, path: Optional[str] = None, 
             param_schema: Optional[Type[BaseModel]] = None,
             response_schema: Optional[Type[BaseModel]] = None,
             retry_config: Optional[Dict[str, Any]] = None):
    """Decorator to mark a method as an API endpoint"""
    def decorator(func):
        func._endpoint_info = {
            'name': name,
            'path': path,
            'param_schema': param_schema,
            'response_schema': response_schema,
            'retry_config': retry_config,
            'description': func.__doc__
        }
        return func
    return decorator

# --- Response Processing ---
class ResponseProcessor:
    """Process API responses into a standardized format"""
    
    @staticmethod
    def process(response: Any, schema: Optional[Type[BaseModel]] = None) -> Dict[str, Any]:
        """Process an API response into a standardized format"""
        # Handle non-dict responses
        if not isinstance(response, dict):
            try:
                # Try to parse JSON strings
                if isinstance(response, str):
                    response = json.loads(response)
                else:
                    return {
                        "status": ApiResponseStatus.ERROR,
                        "errors": [{"code": "INVALID_RESPONSE", "message": f"Unexpected response type: {type(response)}"}]
                    }
            except Exception as e:
                return {
                    "status": ApiResponseStatus.ERROR,
                    "errors": [{"code": "PARSE_ERROR", "message": f"Could not parse response: {str(e)}"}]
                }
        
        # If schema is provided, validate against it
        if schema:
            try:
                validated = schema(**response)
                # Return as dict but keep the original response too
                return {"validated": validated.dict(), "original": response}
            except Exception as e:
                # Just log the validation error but continue with processing
                logger.warning(f"Response validation failed: {str(e)}")
        
        processed_response = {}
        
        # Determine and standardize response status
        if "status" in response:
            processed_response["status"] = response["status"]
        elif "success" in response:
            processed_response["status"] = ApiResponseStatus.SUCCESS if response["success"] else ApiResponseStatus.ERROR
        elif "errors" in response or "error" in response:
            processed_response["status"] = ApiResponseStatus.ERROR
        else:
            processed_response["status"] = ApiResponseStatus.SUCCESS
            
        # Process errors if present
        if processed_response["status"] == ApiResponseStatus.ERROR or "errors" in response or "error" in response:
            error_info = response.get("errors") or response.get("error")
            processed_response["errors"] = ResponseProcessor._standardize_errors(error_info)
            
        # Process data based on common patterns
        for data_field in ["results", "data", "items", "content", "response"]:
            if data_field in response:
                data = response[data_field]
                processed_response["data"] = data
                
                # Additional processing for large data sets
                if isinstance(data, list) and len(data) > 5:
                    processed_response["data_summary"] = {
                        "count": len(data),
                        "samples": data[:3],
                    }
                    
                    # Try to extract common fields for summary
                    if len(data) > 0 and isinstance(data[0], dict):
                        field_counts = {}
                        for item in data:
                            for field in item.keys():
                                field_counts[field] = field_counts.get(field, 0) + 1
                                
                        # Fields that appear in at least 50% of items
                        common_fields = [field for field, count in field_counts.items() 
                                        if count >= len(data) * 0.5]
                        
                        processed_response["data_summary"]["common_fields"] = common_fields
                break  # Only process the first matching data field
        
        # Handle pagination metadata
        for meta_field in ["metadata", "meta", "page_info", "pagination"]:
            if meta_field in response:
                meta = response[meta_field]
                # Standardize pagination metadata
                processed_response["metadata"] = ResponseProcessor._standardize_pagination(meta)
                break  # Only process the first matching metadata field
                
        # Include request info if available
        if "request_id" in response:
            processed_response["request_id"] = response["request_id"]
            
        # Include original response
        processed_response["_original"] = response
        
        return processed_response
    
    @staticmethod
    def _standardize_errors(errors) -> List[Dict[str, Any]]:
        """Standardize error formats from different APIs"""
        if errors is None:
            return []
            
        standard_errors = []
        
        if isinstance(errors, list):
            for error in errors:
                if isinstance(error, dict):
                    standard_errors.append({
                        "code": error.get("code", "UNKNOWN_ERROR"),
                        "message": error.get("message", str(error)),
                        "field": error.get("field") or error.get("path"),
                        "details": error.get("details") or error
                    })
                else:
                    standard_errors.append({
                        "code": "UNKNOWN_ERROR",
                        "message": str(error)
                    })
        elif isinstance(errors, dict):
            standard_errors.append({
                "code": errors.get("code", "UNKNOWN_ERROR"),
                "message": errors.get("message", str(errors)),
                "field": errors.get("field") or errors.get("path"),
                "details": errors
            })
        else:
            standard_errors.append({
                "code": "UNKNOWN_ERROR",
                "message": str(errors)
            })
            
        return standard_errors
        
    @staticmethod
    def _standardize_pagination(metadata) -> Dict[str, Any]:
        """Standardize pagination metadata from different APIs"""
        if not isinstance(metadata, dict):
            return {"raw": metadata}
            
        standard_metadata = {}
        
        # Extract common pagination fields with different naming conventions
        # Total items/results
        for field in ["total", "total_count", "totalCount", "count", "total_items"]:
            if field in metadata:
                standard_metadata["total_items"] = metadata[field]
                break
                
        # Current page
        for field in ["page", "current_page", "currentPage", "page_number", "pageNumber"]:
            if field in metadata:
                standard_metadata["page"] = metadata[field]
                break
                
        # Page size
        for field in ["page_size", "pageSize", "limit", "per_page", "perPage"]:
            if field in metadata:
                standard_metadata["page_size"] = metadata[field]
                break
                
        # Next page indicator
        for field in ["has_more", "hasMore", "has_next_page", "hasNextPage"]:
            if field in metadata:
                standard_metadata["has_more"] = bool(metadata[field])
                break
        
        # Next cursor for cursor-based pagination
        for field in ["next_cursor", "nextCursor", "cursor", "after"]:
            if field in metadata:
                standard_metadata["next_cursor"] = metadata[field]
                break
                
        # Include original metadata
        standard_metadata["_original"] = metadata
        
        return standard_metadata

# --- Enhanced API Tool Wrapper ---
class ApiToolWrapper:
    """Wraps an API service endpoint as a LangChain tool"""
    
    def __init__(self, service: ApiService, endpoint_name: str):
        self.service = service
        self.endpoint_name = endpoint_name
        self.endpoint_info = service.get_endpoint(endpoint_name)
        
    def execute(self, **params) -> Dict[str, Any]:
        """Execute the API call with retries and error handling"""
        retry_config = self.endpoint_info.get('retry_config', {})
        max_retries = retry_config.get('max_retries', 3)
        retry_delay = retry_config.get('retry_delay', 1)
        retry_backoff = retry_config.get('retry_backoff', 2)
        
        retries = 0
        while retries <= max_retries:
            try:
                # Make the actual API call
                response = self.service.call_endpoint(self.endpoint_name, **params)
                
                # Process the response
                processed_response = ResponseProcessor.process(
                    response, 
                    self.endpoint_info.get('response_schema')
                )
                
                return processed_response
                
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    return {
                        "status": ApiResponseStatus.ERROR,
                        "errors": [{
                            "code": "API_CALL_FAILED",
                            "message": f"Failed after {max_retries} attempts",
                            "details": str(e)
                        }]
                    }
                
                # Wait before retry (with exponential backoff)
                wait_time = retry_delay * (retry_backoff ** (retries - 1))
                logger.info(f"API call failed, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)

# --- Service Registry ---
class ServiceRegistry:
    """Registry for API services"""
    
    def __init__(self):
        self._services = {}
        
    def register_service(self, service: ApiService):
        """Register an API service"""
        self._services[service.service_name] = service
        
    def get_service(self, service_name: str) -> Optional[ApiService]:
        """Get a service by name"""
        return self._services.get(service_name)
        
    def list_services(self) -> Dict[str, ApiService]:
        """List all registered services"""
        return self._services
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get LangChain tools for all endpoints in all services"""
        tools = []
        
        for service_name, service in self._services.items():
            for endpoint_name in service.get_endpoint_names():
                tools.append(self._create_tool(service, endpoint_name))
                
        # Add the interpret response tool
        tools.append(interpret_api_response)
        
        return tools
        
    def _create_tool(self, service: ApiService, endpoint_name: str) -> BaseTool:
        """Create a LangChain tool for a service endpoint"""
        endpoint_info = service.get_endpoint(endpoint_name)
        tool_wrapper = ApiToolWrapper(service, endpoint_name)
        
        # Create a unique tool name
        tool_name = f"{service.service_name}_{endpoint_name}"
        
        # Create the tool function
        @tool(tool_name)
        def api_tool(**params) -> Dict:
            """Tool function that calls the API endpoint"""
            return tool_wrapper.execute(**params)
        
        # Set the docstring and other attributes
        api_tool.__name__ = tool_name
        api_tool.__doc__ = endpoint_info['description']
        
        return api_tool

# --- Response Interpretation Tool ---
@tool("interpret_response")
def interpret_api_response(response: Dict[str, Any]) -> str:
    """
    Analyze and interpret a complex API response.
    This tool helps break down and understand the structure and meaning of API responses.
    
    Args:
        response: The API response dictionary to interpret
        
    Returns:
        A structured explanation of what the response contains and how to use it
    """
    interpretation = "# API Response Analysis\n\n"
    
    # Handle non-dict responses
    if not isinstance(response, dict):
        return f"Response is not a dictionary but a {type(response)}. Raw value: {str(response)[:200]}"
    
    # Check status
    status = response.get("status")
    if status == ApiResponseStatus.ERROR:
        interpretation += "## Error Response\n\n"
        
        # Process errors
        if "errors" in response:
            interpretation += "**Errors:**\n\n"
            for i, error in enumerate(response["errors"]):
                interpretation += f"{i+1}. **{error.get('code', 'Unknown')}**: {error.get('message', 'No message')}"
                if error.get("field"):
                    interpretation += f" (Field: {error['field']})"
                interpretation += "\n"
            interpretation += "\n"
            
        interpretation += "**Recommended action:** Address the errors before proceeding.\n\n"
        
    elif status == ApiResponseStatus.SUCCESS:
        interpretation += "## Successful Response\n\n"
        
        # Process data
        if "data" in response:
            data = response["data"]
            
            if isinstance(data, list):
                interpretation += f"**Data contains a list with {len(data)} items**\n\n"
                
                if data and isinstance(data[0], dict):
                    # Show fields from first item
                    interpretation += "**Each item contains these fields:**\n"
                    for key in data[0].keys():
                        interpretation += f"- `{key}`\n"
                    interpretation += "\n"
                    
                    # Sample data
                    interpretation += "**Sample item:**\n```json\n"
                    interpretation += json.dumps(data[0], indent=2)
                    interpretation += "\n```\n\n"
                    
            elif isinstance(data, dict):
                interpretation += "**Data contains an object with these fields:**\n"
                for key in data.keys():
                    interpretation += f"- `{key}`\n"
                interpretation += "\n"
                
                # Complete data if small enough
                if len(json.dumps(data)) < 500:
                    interpretation += "**Complete data:**\n```json\n"
                    interpretation += json.dumps(data, indent=2)
                    interpretation += "\n```\n\n"
                    
        # Data summary for large datasets
        if "data_summary" in response:
            summary = response["data_summary"]
            interpretation += f"**Data summary:** Contains {summary.get('count', 0)} items total\n\n"
            
            if "common_fields" in summary:
                interpretation += "**Common fields across items:**\n"
                for field in summary["common_fields"]:
                    interpretation += f"- `{field}`\n"
                interpretation += "\n"
                
        # Pagination info
        if "metadata" in response:
            meta = response["metadata"]
            interpretation += "**Pagination information:**\n"
            
            if "total_items" in meta:
                interpretation += f"- Total items: {meta['total_items']}\n"
            if "page" in meta:
                interpretation += f"- Current page: {meta['page']}\n"
            if "page_size" in meta:
                interpretation += f"- Page size: {meta['page_size']}\n"
            if "has_more" in meta:
                interpretation += f"- Has more pages: {meta['has_more']}\n"
            if "next_cursor" in meta:
                interpretation += f"- Next cursor: {meta['next_cursor']}\n"
                
            interpretation += "\n"
    
    elif status == ApiResponseStatus.PARTIAL:
        interpretation += "## Partial Success Response\n\n"
        interpretation += "This response indicates partial success. Some operations succeeded while others failed.\n\n"
        
        # Include both data and errors
        if "data" in response:
            interpretation += "**Successful parts:**\n"
            interpretation += f"- Contains data of type: {type(response['data'])}\n"
            if isinstance(response['data'], list):
                interpretation += f"- {len(response['data'])} successful items\n"
            interpretation += "\n"
            
        if "errors" in response:
            interpretation += "**Failed parts:**\n"
            interpretation += f"- {len(response['errors'])} errors occurred\n"
            interpretation += "\n"
    
    # Add request ID if available
    if "request_id" in response:
        interpretation += f"**Request ID:** {response['request_id']}\n\n"
        
    interpretation += "**Note:** This is a processed analysis of the API response. The original response may contain additional fields not mentioned here."
    
    return interpretation

# --- Example API Service Implementations ---

# -- User Service --
class UserParams(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")

class UserCreateParams(BaseModel):
    name: str = Field(..., description="User's full name")
    email: str = Field(..., description="User's email address")
    role: str = Field("user", description="User's role (user, admin)")

class UserUpdateParams(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    name: Optional[str] = Field(None, description="User's full name")
    email: Optional[str] = Field(None, description="User's email address")
    role: Optional[str] = Field(None, description="User's role (user, admin)")

class UserResponse(ApiSuccessResponse):
    data: Dict[str, Any] = Field(..., description="User profile data")

class UserService(ApiService):
    """Service for managing users"""
    
    def __init__(self):
        super().__init__("user", "https://api.example.com/v1/users")
    
    @endpoint("get_user", path="/users/{user_id}", param_schema=UserParams)
    def get_user(self, user_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific user.
        
        Args:
            user_id: Unique identifier for the user
                
        Returns:
            User profile data
        """
        # Simulate API call
        time.sleep(0.5)
        
        # Return mock response
        if user_id == "invalid":
            return {
                "status": ApiResponseStatus.ERROR,
                "errors": [{"code": "USER_NOT_FOUND", "message": "User not found"}]
            }
        
        return {
            "status": ApiResponseStatus.SUCCESS,
            "data": {
                "id": user_id,
                "name": f"User {user_id}",
                "email": f"user{user_id}@example.com",
                "account_type": "premium",
                "created_at": "2023-01-15T10:30:00Z"
            }
        }
    
    @endpoint("create_user", path="/users", param_schema=UserCreateParams)
    def create_user(self, name: str, email: str, role: str = "user") -> Dict[str, Any]:
        """
        Create a new user account.
        
        Args:
            name: User's full name
            email: User's email address
            role: User's role (default: user)
                
        Returns:
            Created user profile data
        """
        # Simulate API call
        time.sleep(0.7)
        
        # Check for invalid input
        if '@' not in email:
            return {
                "status": ApiResponseStatus.ERROR,
                "errors": [{"code": "INVALID_EMAIL", "message": "Email address is invalid", "field": "email"}]
            }
        
        # Generate a user ID
        user_id = f"u{int(time.time())}"
        
        return {
            "status": ApiResponseStatus.SUCCESS,
            "data": {
                "id": user_id,
                "name": name,
                "email": email,
                "role": role,
                "created_at": "2025-04-22T10:30:00Z"
            }
        }
    
    @endpoint("update_user", path="/users/{user_id}", param_schema=UserUpdateParams)
    def update_user(self, user_id: str, name: Optional[str] = None, 
                   email: Optional[str] = None, role: Optional[str] = None) -> Dict[str, Any]:
        """
        Update a user's information.
        
        Args:
            user_id: Unique identifier for the user
            name: User's full name (optional)
            email: User's email address (optional)
            role: User's role (optional)
                
        Returns:
            Updated user profile data
        """
        # Simulate API call
        time.sleep(0.5)
        
        # Check if user exists
        if user_id == "invalid":
            return {
                "status": ApiResponseStatus.ERROR,
                "errors": [{"code": "USER_NOT_FOUND", "message": "User not found"}]
            }
        
        # Check for invalid input
        if email is not None and '@' not in email:
            return {
                "status": ApiResponseStatus.ERROR,
                "errors": [{"code": "INVALID_EMAIL", "message": "Email address is invalid", "field": "email"}]
            }
        
        # Mock existing user data
        existing_user = {
            "id": user_id,
            "name": f"User {user_id}",
            "email": f"user{user_id}@example.com",
            "role": "user",
            "created_at": "2023-01-15T10:30:00Z"
        }
        
        # Update the fields
        if name:
            existing_user["name"] = name
        if email:
            existing_user["email"] = email
        if role:
            existing_user["role"] = role
        
        return {
            "status": ApiResponseStatus.SUCCESS,
            "data": existing_user
        }

# -- Search Service --
class SearchParams(BaseModel):
    query: str = Field(..., description="Search query string")
    limit: int = Field(10, description="Maximum number of results to return")
    page: int = Field(1, description="Page number for pagination")
    sort_by: Optional[str] = Field(None, description="Field to sort results by")
    
class FileSearchParams(BaseModel):
    query: str = Field(..., description="Search query string")
    file_type: Optional[str] = Field(None, description="Filter by file type (pdf, doc, etc.)")
    limit: int = Field(10, description="Maximum number of results to return")

class SearchService(ApiService):
    """Service for search functionality"""
    
    def __init__(self):
        super().__init__("search", "https://api.example.com/v1/search")
    
    @endpoint("search", path="/general", param_schema=SearchParams)
    def search(self, query: str, limit: int = 10, page: int = 1, 
              sort_by: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for information across the knowledge base.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return (default: 10)
            page: Page number for pagination (default: 1)
            sort_by: Field to sort results by (optional)
                
        Returns:
            Results matching the search query
        """
        # Simulate API call
        time.sleep(0.5)
        
        # Calculate offset based on page and limit
        offset = (page - 1) * limit
        
        # Total mock results
        total_results = 42
        
        # Generate mock results
        results = []
        for i in range(min(limit, total_results - offset)):
            result_id = offset + i + 1
            results.append({
                "id": f"r{result_id}",
                "title": f"Result {result_id} for '{query}'",
                "description": f"This is a description for result {result_id} matching '{query}'",
                "score": round(0.95 - (0.01 * i), 2),
                "category": "article" if i % 2 == 0 else "document"
            })
        
        return {
            "status": ApiResponseStatus.SUCCESS,
            "data": results,
            "metadata": {
                "total_count": total_results,
                "page": page,
                "page_size": limit,
                "query": query,
                "has_more": (offset + limit) < total_results
            }
        }
    
    @endpoint("search_files", path="/files", param_schema=FileSearchParams)
    def search_files(self, query: str, file_type: Optional[str] = None, 
                    limit: int = 10) -> Dict[str, Any]:
        """
        Search specifically for files matching criteria.
        
        Args:
            query: Search query string
            file_type: Filter by file type (pdf, doc, etc.) (optional)
            limit: Maximum number of results to return (default: 10)
                
        Returns:
            File results matching the search query
        """
        # Simulate API call
        time.sleep(0.6)
        
        # Generate mock results
        results = []
        total_results = 15
        
        for i in range(min(limit, total_results)):
            # Default file type if none specified
            result_file_type = file_type or ("pdf" if i % 3 == 0 else "doc" if i % 3 == 1 else "xls")
            
            # If file_type is specified, only include matching files
            if file_type and result_file_type != file_type:
                continue
                
            results.append({
                "id": f"f{i+1}",
                "name": f"File-{i+1}-{query}.{result_file_type}",
                "file_type": result_file_type,
                "size": (i+1) * 1024 * 1024,  # Size in bytes
                "created_at": "2025-03-15T10:30:00Z",
                "download_url": f"https://example.com/files/f{i+1}"

```
