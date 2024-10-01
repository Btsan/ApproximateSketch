Estimates are stored in CSV format, with their corresponding subquery, annotated with their parent query in JOB-light(-ranges), actual cardinality, and number of tables.

A variety of estimates (and their Q-error) are stored, but columns labeled `CMM` (Count-Mean-Min) and `COMPASS` are particularly noted in our work.
Methods prefixed `approx` or `upper` refer to our approximated sketches, otherwise they are the exact sketches.
E.g., the column labeled `upper_cmm` is the cardinality estimate using our upper-bounded approximation of Count-Mean-Min sketches.
