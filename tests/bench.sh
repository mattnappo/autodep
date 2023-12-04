#!/bin/bash

ali --duration=10s \
    --rate=10 \
    --workers=8 \
    --header 'Content-Type: application/json' \
    --body-file=./img2img_req.json \
    --method=POST http://localhost:9000/inference

