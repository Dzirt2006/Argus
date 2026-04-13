# Home Assistant Setup

Quick reference for getting HA running and paired with TP-Link (Kasa) devices.

---

## 1. Run HA

HA runs via `docker-compose.yml` with `network_mode: host` — it binds directly to the host's network stack so LAN device discovery (mDNS, UDP broadcast) works.

```bash
docker compose up -d homeassistant
```

Config lives at `./ha_config/`. First boot creates `.storage/` with the admin account set up through the onboarding wizard.

UI: `http://<host-lan-ip>:8123`

---

## 2. VPN caveats

Because HA uses host networking, it inherits the host's routes and interfaces. Two things to know:

- **LAN discovery breaks under full-tunnel VPN.** Kasa uses UDP broadcast for auto-discovery. If the default route is pushed through a VPN, broadcasts leave the wrong interface and devices never respond. **Disconnect the VPN before pairing devices.** Once paired, HA talks to them unicast by IP and survives VPN fine (assuming split-tunnel for the LAN subnet).
- **HA ports are exposed on every interface,** including VPN. If you're on a work VPN, `:8123` is reachable from the corporate network. Firewall or drop inbound on the VPN interface:
  ```
  sudo iptables -I INPUT -i <vpn-iface> -p tcp -m multiport \
    --dports 8000,8001:8006,8080,8123 -j DROP
  ```

---

## 3. Add TP-Link (Kasa) devices

The integration is named **"TP-Link Smart Home"**, not "Kasa". Searching "Kasa" only surfaces *Emulated Kasa*, which is a different thing (makes HA pretend to *be* a Kasa device for legacy Alexa).

1. Settings → Devices & Services → **+ Add Integration** → search `TP-Link`
2. HA auto-discovers devices on the LAN. Accept the prompt.
3. **Authentication prompt:** modern Kasa firmware (post ~2022) uses the KLAP protocol, which requires your TP-Link cloud account credentials. Use the same email/password as the Kasa Smart app.
   - Credentials are used as a pre-shared secret for a **local** ECDH handshake — they are NOT sent to TP-Link servers. Traffic stays on the LAN.
   - Email is case-sensitive.
   - If the device rejects the challenge, the password in the Kasa cloud has drifted from the hash the device cached at provisioning time. Fix: remove and re-add the device in the Kasa app, then retry in HA.

Each device shows up as one or more entities (`switch.*`, `light.*`). Test with Developer Tools → Services → `switch.toggle`.

---

## 4. Disable Bluetooth integration

HA auto-loads the Bluetooth integration if it sees a host adapter. Under host networking with a flaky adapter this spams the log with:

```
habluetooth.scanner ... Failed to force stop scanner: 'NoneType' object has no attribute 'send'
```

Kasa and Nanoleaf are WiFi — BT isn't needed. Remove it:

Settings → Devices & Services → Bluetooth card → ⋮ → **Delete**.

---

## 5. Expose entities + create access token for the agent

HA ships an MCP server integration that the Argus agent connects to.

1. **Add MCP server:** Settings → Devices & Services → + Add Integration → search **"Model Context Protocol Server"** → add with defaults.
2. **Whitelist entities:** Settings → Voice Assistants → Expose → check only the entities the agent should control (lights, switches). Default-deny everything else — the LLM should not have access to unrelated automations.
3. **Create long-lived access token:** profile avatar (bottom-left) → Security tab → **Long-lived access tokens** → Create → name it `argus-agent` → copy the token once (not shown again).
4. **Stash in `.env`:**
   ```
   HA_URL=http://<host-lan-ip>:8123
   HA_TOKEN=<paste>
   ```

The agent's `MultiServerMCPClient` config will reference these when connecting to `${HA_URL}/mcp_server/sse`.
