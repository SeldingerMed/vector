# Lumen forecast: action-conditioned penetration prediction

You receive 10 steps of observation/action history plus the 20-step future
action sequence actually executed. Forecast penetration (`max_pen`, mm) and
the `unsafe` flag at horizons 1, 5, and 20 of that executed future.

Return `{"forecast": {"h1": {"max_pen": f, "unsafe": b}, "h5": {...}, "h20": {...}}}`.

The oracle is the recorded future under the executed actions only. There is
no ground truth for action sequences that were not taken.
