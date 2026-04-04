"""Google Calendar MCP server.

First run: will open a browser for OAuth consent. Token is saved to
/data/calendar_token.json for subsequent runs.

Tools:
  - get_upcoming_events: list events in the next N days
  - search_events: search for events by keyword
  - create_event: create a new calendar event
"""

import datetime as dt
import json
from pathlib import Path

from fastmcp import FastMCP
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

mcp = FastMCP("calendar")

_SCOPES = ["https://www.googleapis.com/auth/calendar"]
_TOKEN_PATH = Path("/data/calendar_token.json")
_CREDENTIALS_PATH = Path("/data/calendar_credentials.json")


def _get_service():
    """Build and return an authenticated Google Calendar API service."""
    creds = None

    if _TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not _CREDENTIALS_PATH.exists():
                raise FileNotFoundError(
                    f"Google OAuth credentials not found at {_CREDENTIALS_PATH}. "
                    "Download from Google Cloud Console → APIs & Services → Credentials → "
                    "OAuth 2.0 Client ID → Download JSON, and save to /data/calendar_credentials.json"
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(_CREDENTIALS_PATH), _SCOPES
            )
            creds = flow.run_local_server(port=0)

        _TOKEN_PATH.write_text(creds.to_json())

    return build("calendar", "v3", credentials=creds)


@mcp.tool()
def get_upcoming_events(days_ahead: int = 7) -> str:
    """Get upcoming events from Google Calendar.

    Use this when the user asks about their schedule, what's coming up,
    or wants to know about appointments in the next few days.

    Args:
        days_ahead: Number of days to look ahead (1-30, default 7).
    """
    days_ahead = max(1, min(30, days_ahead))
    service = _get_service()

    now = dt.datetime.now(dt.timezone.utc)
    end = now + dt.timedelta(days=days_ahead)

    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=end.isoformat(),
            maxResults=20,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = result.get("items", [])
    if not events:
        return f"No events in the next {days_ahead} days."

    lines = [f"**Upcoming events ({days_ahead} days):**\n"]
    for e in events:
        start = e["start"].get("dateTime", e["start"].get("date"))
        end_t = e["end"].get("dateTime", e["end"].get("date"))
        summary = e.get("summary", "(no title)")
        location = e.get("location", "")

        line = f"- **{summary}** — {start}"
        if location:
            line += f" @ {location}"
        lines.append(line)

    return "\n".join(lines)


@mcp.tool()
def search_events(query: str, days_ahead: int = 30) -> str:
    """Search Google Calendar events by keyword.

    Use this when the user asks about a specific event, meeting, or
    appointment by name or topic.

    Args:
        query: Search term — matches against event title, description, location.
               Example: "dentist" or "team standup"
        days_ahead: How far ahead to search (1-90, default 30).
    """
    days_ahead = max(1, min(90, days_ahead))
    service = _get_service()

    now = dt.datetime.now(dt.timezone.utc)
    end = now + dt.timedelta(days=days_ahead)

    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=end.isoformat(),
            q=query,
            maxResults=10,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = result.get("items", [])
    if not events:
        return f"No events matching '{query}' in the next {days_ahead} days."

    lines = [f"**Events matching '{query}':**\n"]
    for e in events:
        start = e["start"].get("dateTime", e["start"].get("date"))
        summary = e.get("summary", "(no title)")
        lines.append(f"- **{summary}** — {start}")

    return "\n".join(lines)


@mcp.tool()
def create_event(
    title: str,
    start: str,
    end: str,
    description: str = "",
    location: str = "",
) -> str:
    """Create a new event on Google Calendar.

    Use this when the user asks to schedule something, add a meeting,
    or create a reminder on their calendar.

    Args:
        title: Event title. Example: "Dentist appointment"
        start: Start time in ISO 8601 format. Example: "2026-03-28T14:00:00"
        end: End time in ISO 8601 format. Example: "2026-03-28T15:00:00"
        description: Optional event description or notes.
        location: Optional location. Example: "123 Main St"
    """
    service = _get_service()

    event = {
        "summary": title,
        "start": {"dateTime": start, "timeZone": "America/Chicago"},
        "end": {"dateTime": end, "timeZone": "America/Chicago"},
    }
    if description:
        event["description"] = description
    if location:
        event["location"] = location

    created = service.events().insert(calendarId="primary", body=event).execute()

    return (
        f"Event created: **{created['summary']}**\n"
        f"When: {created['start']['dateTime']}\n"
        f"Link: {created.get('htmlLink', 'n/a')}"
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8005)
