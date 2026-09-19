import asyncio
import json
import websockets


async def main():
    uri = "ws://localhost:8000/ws/experiments/3"

    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")

        while True:
            message = await websocket.recv()
            metric = json.loads(message)

            print(
                f"Step {metric['step']:4d} | "
                f"Train {metric['train_loss']:.4f} | "
                f"Val {metric['val_loss']:.4f} | "
                f"TPS {metric['tokens_per_sec']:.0f}"
            )


asyncio.run(main())