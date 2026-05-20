# Media MCP — host audio setup

The `media` MCP server plays YouTube Music in a container via `mpv`. Because
the container has no audio hardware of its own, you must expose the host's
audio stack to it. PulseAudio is the default path; ALSA works as a fallback
and is documented as advanced/unsupported.

## PulseAudio (default)

The `media` service in `docker-compose.yml` mounts the user's PulseAudio
socket and authentication cookie into the container. Out of the box it
assumes:

- Host UID is `1000`
- `XDG_RUNTIME_DIR` is `/run/user/1000`
- `~/.config/pulse/cookie` exists

If any of those don't match your system, override them in `.env`:

```bash
PUID=1001
PULSE_SERVER=unix:/run/user/1001/pulse/native
```

You can verify the host side is healthy with:

```bash
pactl info        # should print a Server String matching PULSE_SERVER
ls "${XDG_RUNTIME_DIR}/pulse/native"
```

After `docker compose up media`, check that mpv inside the container can
reach the host:

```bash
docker compose exec media mpv --no-video --length=1 /dev/zero
```

If you hear silence and see `[ao/pulse] Init failed`, the socket/cookie
mount is wrong — re-check `PULSE_SERVER` and the `XDG_RUNTIME_DIR` mount.

## ALSA (advanced/unsupported)

If you don't run PulseAudio, swap the `media` service's `environment` and
`volumes` blocks for:

```yaml
    devices:
      - /dev/snd
    group_add:
      - audio
```

This grants raw access to ALSA devices. Use only on dedicated hosts — it
bypasses Pulse mixing, so any other audio user will get device-busy errors.

## Verifying end-to-end

With the stack up:

```bash
docker compose exec agent python -m agent
# then ask: "play some lofi"
```

You should hear audio from the host's default sink. If the agent reports
"Playback started" but no sound comes out, audio passthrough is the issue —
not the MCP server itself. Run the `mpv --length=1 /dev/zero` check above.

## Why this is fiddly

Container audio depends on the host's audio stack, the host UID, and the
container user matching up. There's no universal one-liner. The defaults
here cover the common single-user Linux desktop case (UID 1000, PulseAudio,
default cookie path). Anything else needs manual override.
