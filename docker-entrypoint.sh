#!/bin/bash

# If running in Hugging Face Spaces (SPACE_ID is set)
if [ -n "$SPACE_ID" ]; then
    export NEXT_PUBLIC_API_URL="https://$SPACE_ID.hf.space"
else
    # Local development
    export NEXT_PUBLIC_API_URL="http://localhost:7860"
fi

# Start supervisor
exec supervisord -c /etc/supervisor/conf.d/supervisord.conf 