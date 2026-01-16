import os.path
import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Scopes required for the Service Account
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Use the calendar ID from ENV or default to primary
CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")

def get_calendar_service():
    """Authenticates using a Service Account and returns the service object."""
    if not os.path.exists("service_account.json"):
        raise FileNotFoundError("service_account.json not found. Please provide the Service Account key.")

    creds = service_account.Credentials.from_service_account_file(
        "service_account.json", scopes=SCOPES
    )
    return build("calendar", "v3", credentials=creds)

def list_upcoming_events(max_results=10):
    """Lists the next upcoming events on the calendar."""
    try:
        service = get_calendar_service()
        now = datetime.datetime.utcnow().isoformat() + "Z" 
        
        events_result = (
            service.events()
            .list(
                calendarId=CALENDAR_ID,
                timeMin=now,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])
        if not events: return "No upcoming events found."

        results = []
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            results.append(f"{start} - {event.get('summary')}")
        return "\n".join(results)

    except HttpError as error:
        logger.error(f"Calendar API error: {error}")
        if error.resp.status == 404:
            return f"Error: Calendar '{CALENDAR_ID}' not found. Please ensure it is shared with the service account email."
        return f"Error connecting to Google Calendar: {error}"

def create_event(summary, start_time_iso, end_time_iso, description="", attendee_email=None):
    """Creates a calendar event in the background."""
    try:
        service = get_calendar_service()
        
        event = {
            'summary': summary,
            'description': description,
            'start': {
                'dateTime': start_time_iso,
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_time_iso,
                'timeZone': 'UTC',
            },
        }
        
        if attendee_email:
            event['attendees'] = [{'email': attendee_email}]

        try:
            event = service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
        except HttpError as error:
            # Handle "Service accounts cannot invite attendees" error (403)
            # We retry without the attendee, but add them to description
            if error.resp.status == 403 and "Service accounts cannot invite attendees" in str(error):
                logger.warning("Service account cannot invite attendees. Retrying without attendee list.")
                if 'attendees' in event:
                    del event['attendees']
                    # Append to description
                    event['description'] = f"{event.get('description', '')}\n\nAttendee: {attendee_email}".strip()
                event = service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
            else:
                raise error

        logger.info(f"Event created in background: {event.get('htmlLink')}")
        return event  # Return the full event dict, let agent handle the message

    except HttpError as error:
        logger.error(f"Calendar API error: {error}")
        if error.resp.status == 404:
            return f"Error: Calendar '{CALENDAR_ID}' not found. Please ensure it is shared with the service account email."
        return f"Error booking event: {error}"

if __name__ == "__main__":
    try:
        print(list_upcoming_events())
    except Exception as e:
        print(f"Error: {e}")
