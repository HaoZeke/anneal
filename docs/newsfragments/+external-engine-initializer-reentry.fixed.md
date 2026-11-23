<!-- Copyright 2023--present rgpot developers -->

Generic and external engine initializers may load another engine without deadlocking the process. Engine handles enter the resident registry only after their initializer, required-symbol checks, ABI validation and first successful construction complete.
