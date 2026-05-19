# Google Calendar MCP setup

The `calendar` MCP server talks to the Google Calendar API on behalf of your
Google account. It needs an OAuth client secret the first time it runs, and
caches a refresh token afterwards.

The container is headless and the OAuth loopback redirect port isn't exposed,
so the OAuth dance runs **on the host** once. The container then starts with
the cached token and never needs a browser.

## 1. Create OAuth credentials

1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Create (or reuse) a project, then enable the **Google Calendar API**:
   APIs & Services -> Library -> "Google Calendar API" -> Enable.
3. Configure the OAuth consent screen (External, Testing is fine; add your own
   Google account as a test user).
4. APIs & Services -> Credentials -> Create credentials -> **OAuth client ID** ->
   Application type: **Desktop app**.
5. Click **Download JSON** on the new client. Save it as:

   ```
   ./data/calendar_credentials.json
   ```

   That host path is mounted into the calendar container as
   `/data/calendar_credentials.json`, which is where
   `mcp_servers/calendar/server.py` expects it.

## 2. Bootstrap the OAuth token on the host

From the repo root:

```bash
pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
python -c "from mcp_servers.calendar.server import _get_service; _get_service()"
```

`_get_service()` opens your browser, you approve consent, and the resulting
refresh token is written to `./data/calendar_token.json`.

## 3. Start the container

```bash
docker compose up calendar
```

The container reads the cached `calendar_token.json`, refreshes it
automatically when it expires, and never prompts for a browser.

## Notes

- The token grants the `https://www.googleapis.com/auth/calendar` scope (full
  read/write on the primary calendar). `create_event` is in `DESTRUCTIVE_TOOLS`,
  so the agent will ask for confirmation before writing.
- To revoke access, delete `./data/calendar_token.json` and revoke the app
  under [Google account permissions](https://myaccount.google.com/permissions).
- **Alternative (advanced, unsupported): in-container OAuth.** You'd need to
  expose the OAuth loopback redirect port from the calendar container and
  point a host browser at it on first run. The host-bootstrap flow above is
  simpler and is the recommended path.
