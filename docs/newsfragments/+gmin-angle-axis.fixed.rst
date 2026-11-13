The interleaved TIP4P converters keep GMIN's angle-axis vectors as they are: GMIN's ``NTIP`` places sites by the same active map as the rigid chart, and its ``CHECKD`` energy of a converted start is now anneal's.
``binary_water_campaign`` reads optbench ``.aa`` starts as GMIN ``coords`` (centres, then angle-axis vectors); ``--legacy-interleaved`` keeps the earlier reading, and ``GminAngleAxisFile::LegacyInterleaved`` loads interleaved files written with negated rotations.
``scripts/optbench/anneal_globalopt.py`` opens the archive start GMIN opens.
