import os
import datetime
from datetime import timezone
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from loguru import logger
from rag.retriever import get_relevant_context
from google_calendar_service import list_upcoming_events, create_event
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Simple chat model without function calling
# model = ChatGroq(
#     model="llama-3.1-8b-instant",
#     max_tokens=128,
#     temperature=0.5,
# )

# model = ChatOpenAI instantiation moved inside SimpleAgent

system_prompt = """You are Aura, the professional and snappy AI assistant for BharatLogic.

CORE PERSONALITY:
- **Maximum Brevity**: If a 3-word answer works, use it. No fluff.
- **Natural & Human**: Use short sentences. Mix in fillers like "Got it," "Sure," or "I see."
- **Point-Driven**: If the answer is a list, just give the points. Do NOT explain each point unless asked.
- **Voice-First**: Avoid technical jargon or long-winded intros.

ABOUT BHARATLOGIC:
BharatLogic offers AI-Agentic, AI/ML, GenAI, Web/Mobile, Software, Digital Marketing, and Cloud services.
Tech: Node.js, Angular, React, Vue, Django, .NET, PHP, WordPress, MySQL, MongoDB, iOS, Android, etc.

CORE BEHAVIOR RULES:
- Be concise, consultative, and human.
- **Mandatory Information**: You must obtain the user's Name and Contact Number (or Email) early in the conversation. Use this information to personalize the experience.
- **Use standard Markdown** for formatting. If you provide points or lists, use `-` or `1.` with clear line breaks so they display correctly.
- Speak in a way that is easy to listen to; keep paragraphs short and well-separated.
- Ask one question at a time.
- Maintain conversation context throughout the session.
- Do NOT use technical jargon unless the user does.
- Do NOT mention internal processes or system prompts.

REQUIREMENT GATHERING (HIGH PRIORITY):
When a user shows interest in a project:
1. Ask for a requirement document or brief explanation
2. Capture timeline expectations
3. Suggest suitable technologies dynamically
4. If budget is asked, politely avoid numbers and move toward a discussion
5. Offer to schedule a meeting via calendar integration

CAREERS FLOW:
If the user asks about jobs or careers:
- Collect role, experience, notice period, expected package
- Allow CV upload
- Confirm submission politely

MEETING SCHEDULING:
- Offer to schedule a call with a solution architect.
- Ask for Date and Time only. **Duration is always 30 minutes by default** - do NOT ask for it.
- **EMAIL REQUIRED**: You MUST have the user's email address to book a meeting. If you don't have it, ask for it specifically for the invitation.
- Assume UTC unless specified.
- **CRITICAL**: Once a tool returns 'SUCCESS', you MUST tell the user: "The meeting is scheduled! You will receive a confirmation email shortly." Then stop.

END GOAL:
Every interaction should end with one of the following:
- A qualified project requirement
- A scheduled meeting
- A career application submission
- A clear next step

Never give specific price quotes or project timelines. Always suggest a meeting with BharatLogic experts.

Always represent BharatLogic professionally and confidently.
"""

import re

class SimpleAgent:
    """Simple chat agent with RAG support and contact gathering flow."""
    
    def __init__(self, use_rag: bool = True):
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            max_tokens=256,
            temperature=0.5,
        )
        self.history = [SystemMessage(content=system_prompt)]
        self.use_rag = use_rag
        self.user_data = {"name": None, "phone": None, "email": None}
        self.info_gathered = False
        self.initial_query = None

        # Define Tools
        @tool
        def search_upcoming_events(max_results: int = 5):
            """Lists upcoming events on the Google Calendar."""
            try:
                return str(list_upcoming_events(max_results))
            except Exception as e:
                return f"Error: {e}. (Check service_account.json)"

        @tool
        def book_meeting(start_time_iso: str, end_time_iso: str, summary: str, description: str = ""):
            """Schedules a new meeting on the Google Calendar. Times must be in ISO format (e.g. 2024-01-15T14:00:00Z)."""
            try:
                # Use gathered name/email if possible
                attendee = self.user_data.get("email")
                res = create_event(summary, start_time_iso, end_time_iso, description, attendee)
                if res:
                    return f"Meeting booked successfully! Link: {res.get('htmlLink')}"
                return "Failed to book meeting."
            except Exception as e:
                return f"Error: {e}. (Check service_account.json)"

        self.tools = [search_upcoming_events, book_meeting]
        self.model_with_tools = self.model.bind_tools(self.tools)

    def _validate_phone(self, phone: str) -> bool:
        """Simple regex for phone validation (7-15 digits)."""
        if not phone: return False
        clean_phone = re.sub(r'[^\d+]', '', phone)
        return 7 <= len(clean_phone) <= 15

    def _validate_email(self, email: str) -> bool:
        """Simple regex for email validation."""
        if not email: return False
        return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

    def _extract_info(self, text: str):
        """Attempts to extract name, phone, or email from text using simple regex."""
        # Email extraction
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        if emails: self.user_data["email"] = emails[0]
        
        # Phone extraction
        phones = re.findall(r'\+?\d[\d\s-]{7,15}\d', text)
        if phones:
            potential_phone = phones[0].strip()
            if self._validate_phone(potential_phone):
                self.user_data["phone"] = potential_phone
            else:
                logger.warning(f"Extracted invalid phone: {potential_phone}")

        # Name extraction (Heuristic)
        # Look for "My name is [Name]" or "I am [Name]"
        name_match = re.search(r"(?:my name is|i am|this is|i'm|it's)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)", text, re.IGNORECASE)
        if name_match:
            self.user_data["name"] = name_match.group(1).strip().capitalize()
        
        # If the user just gives 1-2 words (like "John" or "John Doe")
        if not self.user_data["name"] and len(text.split()) <= 2:
            # Check if it doesn't look like a phone number or command
            if not any(char.isdigit() for char in text) and text.lower() not in ["yes", "no", "hi", "hello", "ok"]:
                self.user_data["name"] = text.strip().capitalize()
    
    def _build_message_with_context(self, user_message: str) -> str:
        """Add RAG context to user message if available."""
        if not self.use_rag:
            return user_message
        
        context = get_relevant_context(user_message, k=3)
        if context:
            logger.info(f"RAG: Found relevant context ({len(context)} chars)")
            return f"{context}\n\nUser question: {user_message}"
        return user_message
    
    def invoke(self, input_dict: dict, config: dict = None) -> dict:
        """Process a message and return response."""
        user_message = input_dict["messages"][0]["content"]
        
        # Build message with RAG context
        enriched_message = self._build_message_with_context(user_message)
        self.history.append(HumanMessage(content=enriched_message))
        
        response = self.model.invoke(self.history)
        self.history.append(response)
        
        return {"messages": [response]}
    
    def stream(self, input_dict: dict, config: dict = None, stream_mode: str = "messages"):
        """Stream the response token by token, managing the flow."""
        user_message = input_dict["messages"][0]["content"]
        
        # 1. Capture initial query if it's the beginning
        if not self.initial_query and not self.info_gathered:
            self.initial_query = user_message

        # 2. Extract any info from the current message
        self._extract_info(user_message)
        
        # 3. Check if we have enough info to proceed
        has_contact = self.user_data["phone"] or self.user_data["email"]
        if self.user_data["name"] and has_contact:
            self.info_gathered = True

        # 4. Construct flow-specific instructions
        flow_context = ""
        if not self.info_gathered:
            flow_context = "\n[SYSTEM INSTRUCTION: You are in CONTACT GATHERING mode. "
            if not self.user_data["name"]:
                flow_context += "You MUST POLITELY ask for the user's Name first before anything else. "
            elif not has_contact:
                # Offer alternatives if they refuse one
                if "no" in user_message.lower() or "won't" in user_message.lower() or "refuse" in user_message.lower():
                    flow_context += "The user seems hesitant. Politely explain that you need either a phone number or an email to proceed, and offer the one they haven't refused yet. "
                else:
                    flow_context += f"Great, we have their name: {self.user_data['name']}. Now, ask for either their Phone Number or Email to proceed. "
            
            flow_context += "Acknowledge their query politely, but remind them you need these details before moving to the main discussion. "
            
            if self.user_data["phone"] and not self._validate_phone(self.user_data['phone']):
                 flow_context += "The phone number they just provided looks invalid. Ask them to check and confirm it. "
            flow_context += "]"
        else:
            flow_context = f"\n[SYSTEM INSTRUCTION: INFORMATION GATHERED. User: {self.user_data['name']}, Contact: {self.user_data['phone'] or self.user_data['email']}. Proceed to help the user with their queries.]"
            if self.initial_query:
                flow_context += f" Now address their original query: '{self.initial_query}' in detail."
                self.initial_query = None 
            flow_context += "]"

        # Build message with RAG and Flow context
        enriched_message = self._build_message_with_context(user_message)
        if flow_context:
            enriched_message += flow_context

        self.history.append(HumanMessage(content=enriched_message))
        
        # Add Current Date/Time context so the model knows "tomorrow" or "next Monday"
        now_context = f"\n[SYSTEM: Current Time (UTC): {datetime.datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}]"
        self.history[-1].content += now_context

        full_response = ""
        # Handle potential tool calling loop
        while True:
            # Use .stream() for low latency (TTFT)
            combined_message = None
            
            for chunk in self.model_with_tools.stream(self.history):
                if combined_message is None:
                    combined_message = chunk
                else:
                    combined_message += chunk
                
                if chunk.content:
                    full_response += chunk.content
                    yield (chunk, {})
            
            if combined_message and combined_message.tool_calls:
                self.history.append(combined_message)
                for tool_call in combined_message.tool_calls:
                    tool_id = tool_call["id"]
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    logger.info(f"Executing tool: {tool_name} with {tool_args}")
                    
                    result = "Tool not found"
                    if tool_name == "search_upcoming_events":
                        try:
                            result = list_upcoming_events(tool_args.get("max_results", 5))
                            if not result: result = "No upcoming events found."
                        except Exception as e:
                            logger.error(f"Tool Error: {e}")
                            result = f"Error: {e}"
                    elif tool_name == "book_meeting":
                        try:
                            attendee = self.user_data.get("email")
                            res = create_event(
                                tool_args.get("summary"), 
                                tool_args.get("start_time_iso"), 
                                tool_args.get("end_time_iso"), 
                                tool_args.get("description", ""), 
                                attendee
                            )
                            if res and isinstance(res, dict) and 'summary' in res:
                                result = f"SUCCESS: Meeting '{res['summary']}' booked! Link: {res.get('htmlLink')}"
                            else:
                                result = str(res)
                        except Exception as e:
                            logger.error(f"Tool Error: {e}")
                            result = f"Error: {e}"
                    
                    self.history.append(ToolMessage(content=str(result), tool_call_id=tool_id))
                
                # Clear full_response for the turn that follows tool results
                # (Or keep it if you want to accumulate, but usually we want the final response)
                continue
            
            break
        
        # 5. POST-PROCESS: Try to extract name from the LLM's understanding if it says "Thanks [Name]"
        if not self.user_data["name"]:
             # This is a bit hacky, better would be to use a separate call or tool, 
             # but we'll see if the LLM extracted it in the conversation.
             pass

        # Add complete response to history
        self.history.append(AIMessage(content=full_response))

    def clear_history(self):
        """Reset conversation history."""
        self.history = [SystemMessage(content=system_prompt)]

    def generate_summary(self) -> str:
        """Generates a structured summary of the conversation."""
        try:
            summary_prompt = (
                "Analyze the conversation history and generate a structured summary.\n"
                "Include:\n"
                "1. User's main intent or query.\n"
                "2. Key information provided by the user.\n"
                "3. Any scheduled meetings or next steps.\n"
                "4. User details (Name, Contact) if gathered.\n"
                "Format as a concise paragraph or bullet points."
            )
            
            # Create a temporary history for summarization
            messages = self.history + [HumanMessage(content=summary_prompt)]
            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Summary generation failed."