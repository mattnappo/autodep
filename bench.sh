#!/bin/bash

ali --duration=10s --rate=20 --body-file=bench.json --method=POST http://localhost:9000/image_inference
