import os
import json
import time
import uuid
import httpx
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Ki2API - Claude Sonnet 4 Claude AI Compatible API",
    description="Simple Docker-ready Claude AI-compatible API for Claude Sonnet 4",
    version="1.0.0"
)

# Configuration
API_KEY = os.getenv("API_KEY", "ki2api-key-2024")
KIRO_ACCESS_TOKEN = os.getenv("KIRO_ACCESS_TOKEN")
KIRO_REFRESH_TOKEN = os.getenv("KIRO_REFRESH_TOKEN")
KIRO_BASE_URL = "https://codewhisperer.us-east-1.amazonaws.com/generateAssistantResponse"
PROFILE_ARN = "arn:aws:codewhisperer:us-east-1:699475941385:profile/EHGA3GRVQMUK"

# Model mapping
MODEL_NAME = "claude-sonnet-4-20250514"
CODEWHISPERER_MODEL = "CLAUDE_SONNET_4_20250514_V1_0"


# Pydantic models
class ContentPart(BaseModel):
    type: str = "text"
    text: str

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]
    
    def get_content_text(self) -> str:
        """Extract text content from either string or content parts"""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            # Join all text parts
            text_parts = []
            for part in self.content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and "text" in part:
                        text_parts.append(part["text"])
                elif hasattr(part, 'text'):
                    text_parts.append(part.text)
            return "".join(text_parts)
        return str(self.content)

class ClaudeMessageRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    system: Optional[str] = None
    temperature: Optional[float] = 0.7

class ClaudeMessageResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    type: str = "message"
    role: str = "assistant"
    content: List[Dict[str, Any]]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Dict[str, int]

class ClaudeStreamEvent(BaseModel):
    type: str
    message: Optional[Dict[str, Any]] = None
    index: Optional[int] = None
    content_block: Optional[Dict[str, Any]] = None
    delta: Optional[Dict[str, Any]] = None

# Token management
class TokenManager:
    def __init__(self):
        self.access_token = KIRO_ACCESS_TOKEN
        self.refresh_token = KIRO_REFRESH_TOKEN
        self.refresh_url = "https://prod.us-east-1.auth.desktop.kiro.dev/refreshToken"

    async def refresh_tokens(self):
        if not self.refresh_token:
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.refresh_url,
                    json={"refreshToken": self.refresh_token},
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                self.access_token = data.get("accessToken")
                return self.access_token
        except Exception as e:
            print(f"Token refresh failed: {e}")
            return None

    def get_token(self):
        return self.access_token

token_manager = TokenManager()

# Build CodeWhisperer request
def build_codewhisperer_request(messages: List[ChatMessage], system_prompt: Optional[str] = None):
    conversation_id = str(uuid.uuid4())
    
    # Extract system prompt and user messages
    system_prompt = system_prompt or ""
    user_messages = []
    
    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.get_content_text()
        else:
            user_messages.append(msg)
    
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user messages found")
    
    # Build history
    history = []
    for i in range(0, len(user_messages) - 1, 2):
        if i + 1 < len(user_messages):
            history.append({
                "userInputMessage": {
                    "content": user_messages[i].get_content_text(),
                    "modelId": CODEWHISPERER_MODEL,
                    "origin": "AI_EDITOR"
                }
            })
            history.append({
                "assistantResponseMessage": {
                    "content": user_messages[i + 1].get_content_text(),
                    "toolUses": []
                }
            })
    
    # Build current message
    current_message = user_messages[-1]
    content = current_message.get_content_text()
    if system_prompt:
        content = f"{system_prompt}\n\n{content}"
    
    return {
        "profileArn": PROFILE_ARN,
        "conversationState": {
            "chatTriggerType": "MANUAL",
            "conversationId": conversation_id,
            "currentMessage": {
                "userInputMessage": {
                    "content": content,
                    "modelId": CODEWHISPERER_MODEL,
                    "origin": "AI_EDITOR",
                    "userInputMessageContext": {}
                }
            },
            "history": history
        }
    }

# AWS Event Stream Parser
class AWSStreamParser:
    @staticmethod
    def parse_event_stream_to_json(raw_data: bytes) -> Dict[str, Any]:
        """Parse AWS event stream format to JSON"""
        try:
            # Convert bytes to string if needed
            if isinstance(raw_data, bytes):
                # Try to decode as UTF-8 first
                try:
                    raw_str = raw_data.decode('utf-8')
                except UnicodeDecodeError:
                    # If UTF-8 fails, try to find JSON in binary
                    raw_str = raw_data.decode('utf-8', errors='ignore')
            else:
                raw_str = str(raw_data)
            
            # Look for JSON content in the response
            # AWS event stream contains binary headers followed by JSON payloads
            json_pattern = r'\{[^{}]*"content"[^{}]*\}'
            matches = re.findall(json_pattern, raw_str, re.DOTALL)
            
            if matches:
                content_parts = []
                for match in matches:
                    try:
                        data = json.loads(match)
                        if 'content' in data and data['content']:
                            content_parts.append(data['content'])
                    except:
                        continue
                if content_parts:
                    return {"content": ''.join(content_parts)}
            
            # Try to extract from AWS event stream format
            # Look for :content-type and extract JSON after headers
            content_type_pattern = r':content-type[^:]*:[^:]*:[^:]*:(\{.*\})'
            content_matches = re.findall(content_type_pattern, raw_str, re.DOTALL)
            if content_matches:
                for match in content_matches:
                    try:
                        data = json.loads(match.strip())
                        if isinstance(data, dict) and 'content' in data:
                            return {"content": data['content']}
                    except:
                        continue
            
            # Try to extract any JSON objects
            json_objects = re.findall(r'\{[^{}]*\}', raw_str)
            for obj in json_objects:
                try:
                    data = json.loads(obj)
                    if isinstance(data, dict) and 'content' in data:
                        return {"content": data['content']}
                except:
                    continue
            
            # Final fallback: extract readable text
            readable_text = re.sub(r'[^\x20-\x7E\n\r\t]', '', raw_str)
            readable_text = re.sub(r':event-type[^:]*:[^:]*:[^:]*:', '', readable_text)
            
            # Look for Chinese characters or meaningful content
            chinese_pattern = r'[\u4e00-\u9fff]+'
            chinese_matches = re.findall(chinese_pattern, raw_str)
            if chinese_matches:
                return {"content": ''.join(chinese_matches)}
            
            return {"content": readable_text.strip() or "No content found in response"}
            
        except Exception as e:
            return {"content": f"Error parsing response: {str(e)}"}


# Make API call to Kiro/CodeWhisperer
async def call_kiro_api(messages: List[ChatMessage], system_prompt: Optional[str] = None, stream: bool = False):
    token = token_manager.get_token()
    if not token:
        raise HTTPException(status_code=401, detail="No access token available")
    
    request_data = build_codewhisperer_request(messages, system_prompt)
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                KIRO_BASE_URL,
                headers=headers,
                json=request_data,
                timeout=120
            )
            
            if response.status_code == 403:
                # Try to refresh token
                new_token = await token_manager.refresh_tokens()
                if new_token:
                    headers["Authorization"] = f"Bearer {new_token}"
                    response = await client.post(
                        KIRO_BASE_URL,
                        headers=headers,
                        json=request_data,
                        timeout=120
                    )
            
            response.raise_for_status()
            return response
            
    except Exception as e:
        import traceback
        print(f"API call failed: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=503, detail=f"API call failed: {str(e)}")

# API endpoints
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ki2api"
            }
        ]
    }

@app.post("/v1/messages")
async def create_message(request: ClaudeMessageRequest):
    if request.model != MODEL_NAME:
        raise HTTPException(status_code=400, detail=f"Only {MODEL_NAME} is supported")
    
    if request.stream:
        return await create_streaming_response(request)
    else:
        return await create_non_streaming_response(request)

async def create_non_streaming_response(request: ClaudeMessageRequest):
    response = await call_kiro_api(request.messages, request.system, stream=False)
    return await create_conversion_response(response, request.model)

async def create_conversion_response(response, model: str):
    """Convert AWS event stream to Claude AI format"""
    try:
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        # Get response content as bytes to handle binary data
        response_bytes = response.content
        print(f"Response content type: {type(response_bytes)}")
        print(f"Response content length: {len(response_bytes)}")
        
        # Try to parse as JSON first
        try:
            response_data = response.json()
            print(f"Successfully parsed JSON response")
            if isinstance(response_data, dict) and 'content' in response_data:
                response_text = response_data['content']
            else:
                response_text = str(response_data)
        except Exception as e:
            print(f"JSON parsing failed: {e}")
            # Handle event stream format using AWS parser
            parsed_data = AWSStreamParser.parse_event_stream_to_json(response_bytes)
            response_text = parsed_data.get('content', "")
            print(f"Parsed content length: {len(response_text)}")
            
            if not response_text or response_text == "No content found in response":
                # Last resort: try to decode as text
                try:
                    response_text = response_bytes.decode('utf-8', errors='ignore')
                    print(f"Fallback text decode length: {len(response_text)}")
                except Exception as decode_error:
                    response_text = f"Unable to decode response: {str(decode_error)}"
        
        print(f"Final response text: {response_text[:200]}...")
        
    except Exception as e:
        print(f"Error in conversion: {e}")
        import traceback
        traceback.print_exc()
        response_text = f"Error processing response: {str(e)}"
    
    return ClaudeMessageResponse(
        model=model,
        content=[{
            "type": "text",
            "text": response_text
        }],
        usage={
            "input_tokens": 0,
            "output_tokens": 0
        }
    )


async def create_streaming_response(request: ClaudeMessageRequest):
    response = await call_kiro_api(request.messages, request.system, stream=True)
    return await create_streaming_conversion_response(response, request.model)

async def create_streaming_conversion_response(response, model: str):
    """Convert AWS event stream to Claude AI streaming format"""
    print(f"Starting streaming response, status: {response.status_code}")
    
    async def generate():
        # Send initial response
        initial_event = ClaudeStreamEvent(
            type="message_start",
            message={
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None
            }
        )
        print(f"Sending initial event: {initial_event}")
        yield f"event: message_start\ndata: {json.dumps(initial_event.dict(exclude_none=True))}\n\n"
        
        # Send content block start
        content_block_start_event = ClaudeStreamEvent(
            type="content_block_start",
            index=0,
            content_block={
                "type": "text",
                "text": ""
            }
        )
        yield f"event: content_block_start\ndata: {json.dumps(content_block_start_event.dict(exclude_none=True))}\n\n"
        
        # Read response and stream content
        content = ""
        chunk_count = 0
        
        # Read the entire response as bytes first
        response_bytes = response.content
        print(f"Streaming response bytes length: {len(response_bytes)}")
        
        # Parse the AWS event stream
        try:
            # Convert bytes to string
            if isinstance(response_bytes, bytes):
                response_str = response_bytes.decode('utf-8', errors='ignore')
            else:
                response_str = str(response_bytes)
            
            # Look for content in the AWS event stream
            # AWS uses a specific format with binary headers and JSON payloads
            
            # Method 1: Look for JSON objects with content
            json_pattern = r'\{[^{}]*"content"[^{}]*\}'
            json_matches = re.findall(json_pattern, response_str, re.DOTALL)
            
            if json_matches:
                for match in json_matches:
                    try:
                        data = json.loads(match)
                        if 'content' in data and data['content']:
                            chunk_text = data['content']
                            content += chunk_text
                            chunk_count += 1
                            
                            delta_event = ClaudeStreamEvent(
                                type="content_block_delta",
                                index=0,
                                delta={
                                    "type": "text_delta",
                                    "text": chunk_text
                                }
                            )
                            print(f"Streaming JSON chunk {chunk_count}: {chunk_text[:50]}...")
                            yield f"event: content_block_delta\ndata: {json.dumps(delta_event.dict(exclude_none=True))}\n\n"
                            
                            # Small delay to simulate streaming
                            import asyncio
                            await asyncio.sleep(0.01)
                    except Exception as e:
                        print(f"Error streaming JSON chunk: {e}")
                        continue
            else:
                # Method 2: Try to extract readable text
                readable_text = re.sub(r'[^\x20-\x7E\n\r\t\u4e00-\u9fff]', '', response_str)
                
                # Look for Chinese text specifically
                chinese_pattern = r'[\u4e00-\u9fff][\u4e00-\u9fff\s\.,!?]*[\u4e00-\u9fff]'
                chinese_matches = re.findall(chinese_pattern, response_str)
                
                if chinese_matches:
                    combined_text = ''.join(chinese_matches)
                    # Split into chunks for streaming
                    chunk_size = max(1, len(combined_text) // 10)
                    for i in range(0, len(combined_text), chunk_size):
                        chunk_text = combined_text[i:i+chunk_size]
                        content += chunk_text
                        chunk_count += 1
                        
                        delta_event = ClaudeStreamEvent(
                            type="content_block_delta",
                            index=0,
                            delta={
                                "type": "text_delta",
                                "text": chunk_text
                            }
                        )
                        print(f"Streaming Chinese text chunk {chunk_count}: {chunk_text[:50]}...")
                        yield f"event: content_block_delta\ndata: {json.dumps(delta_event.dict(exclude_none=True))}\n\n"
                        
                        import asyncio
                        await asyncio.sleep(0.05)
                else:
                    # Method 3: Use the entire readable text
                    if readable_text.strip():
                        delta_event = ClaudeStreamEvent(
                            type="content_block_delta",
                            index=0,
                            delta={
                                "type": "text_delta",
                                "text": readable_text.strip()
                            }
                        )
                        print(f"Streaming fallback text: {readable_text[:100]}...")
                        yield f"event: content_block_delta\ndata: {json.dumps(delta_event.dict(exclude_none=True))}\n\n"
                        content = readable_text.strip()
        
        except Exception as e:
            print(f"Error in streaming generation: {e}")
            import traceback
            traceback.print_exc()
            
            # Send error as content
            error_event = ClaudeStreamEvent(
                type="content_block_delta",
                index=0,
                delta={
                    "type": "text_delta",
                    "text": f"Error: {str(e)}"
                }
            )
            yield f"event: content_block_delta\ndata: {json.dumps(error_event.dict(exclude_none=True))}\n\n"
        
        print(f"Streaming complete, total chunks: {chunk_count}, content length: {len(content)}")
        
        # Send content block stop
        content_block_stop_event = ClaudeStreamEvent(
            type="content_block_stop",
            index=0
        )
        yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop_event.dict(exclude_none=True))}\n\n"
        
        # Send message delta
        message_delta_event = ClaudeStreamEvent(
            type="message_delta",
            delta={
                "stop_reason": "end_turn",
                "stop_sequence": None
            }
        )
        yield f"event: message_delta\ndata: {json.dumps(message_delta_event.dict(exclude_none=True))}\n\n"
        
        # Send final response
        message_stop_event = ClaudeStreamEvent(
            type="message_stop"
        )
        yield f"event: message_stop\ndata: {json.dumps(message_stop_event.dict(exclude_none=True))}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "ki2api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8989)