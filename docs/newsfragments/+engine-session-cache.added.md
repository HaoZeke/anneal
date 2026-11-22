<!-- Copyright 2023--present rgpot developers -->

`EngineSession` gains `attach_cache`, `cache_counts` and `clear_cache`, so session users can attach a result cache and read its counts. On Windows the loader refuses a cache path that is not valid Unicode, because the engine opens it as UTF-8.
