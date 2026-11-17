`UmaPot::caps()` reports `batched`.
`UmaPot` already evaluated a batch of images in one AOTInductor call, but the flag was false, so a caller that trusts it, such as a nudged elastic band, evaluated the images one at a time.
