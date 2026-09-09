"""Single source of truth for the app build number.

BUMP THIS on EVERY change to the app (layout, callbacks, data, math). The
header renders it server-side, so after a server restart a browser refresh
always shows the running server's build — if the number on screen doesn't
match this file, the browser is showing a stale page.
"""

APP_BUILD = 35
