# Google Calendar MCP setup

The `calendar` MCP server talks to the Google Calendar API on behalf of your
Google account. It needs an OAuth client secret the first time it runs, and
caches a refresh token afterwards.

## 1. Create OAuth credentials

1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Create (or reuse) a project, then enable the **Google Calendar API**:
   APIs & Services -> Library -> "Google Calendar API" -> Enable.
3. Configure the OAuth consent screen (External, Testing is fine; add your own
   Google account as a test user).
4. APIs & Services -> Credentials -> Create credentials -> **OAuth client ID** ->
   Application type: **Desktop app**.
5. Click **Download JSON** on the new client. This is the `client_secret.json`.

## 2. Drop the file at the right path

Save the downloaded JSON as:

```
./data/calendar_credentials.json
```

(That host path is mounted into the calendar container at
`/data/calendar_credentials.json`, which is where `mcp_servers/calendar/server.py`
expects it.)

## 3. First-run OAuth flow

The first time a calendar tool is invoked, the server runs
`InstalledAppFlow.run_local_server(port=0)`, which opens a browser for consent.
After you approve, the resulting refresh token is persisted to:

```
./data/calendar_token.json
```

Subsequent runs reuse that token (refreshing it automatically when it expires),
so no further browser interaction is needed.

## Notes

- The container needs to be able to open a browser on first run. The easiest
  path is to run the OAuth flow once on the host (or with the container's
  display forwarded), then let Docker reuse the resulting `calendar_token.json`.
- The token grants the `https://www.googleapis.com/auth/calendar` scope
  (full read/write on the primary calendar). `create_event` is in
  `DESTRUCTIVE_TOOLS`, so the agent will ask for confirmation before writing.
- To revoke access, delete `./data/calendar_token.json` and revoke the app
  under [Google account permissions](https://myaccount.google.com/permissions).
