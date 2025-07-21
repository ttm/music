"""Synthesize a short sung phrase using the singing utilities."""

import music

# Clone eCantorix engine if not already installed
music.singing.setup_engine()

# Render a short sung phrase inside the local cache folder
music.singing.make_test_song()
