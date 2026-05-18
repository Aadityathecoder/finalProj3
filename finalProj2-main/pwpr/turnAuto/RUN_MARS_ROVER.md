# DSE Mars Rover Console

Run from this folder:

```bash
python3 -m pip install --user -r requirements.txt
python3 -m uvicorn api:app --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

Use **Start Video Sim** to show the rover camera simulation, path overlay, center line, autonomous navigation, hazard avoidance, and mission-complete Martian life alert.

API checklist:

- `GET /status`
- `GET /mission`
- `POST /power/on`
- `POST /autonomous/start`
- `POST /simulation/start`
- `POST /forward`, `/backward`, `/left`, `/right`, `/stop`
- `PUT /mission`
- `DELETE /mission`
