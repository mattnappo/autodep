#!/bin/bash

ali --duration=10s \
    --rate=10 \
    --workers=8 \
    --method=GET 'http://localhost:9000/workers/status'

