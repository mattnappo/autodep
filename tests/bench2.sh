#!/bin/bash

ali --duration=10s \
    --rate=60 \
    --workers=14 \
    --method=GET 'http://localhost:9000/workers/status'

