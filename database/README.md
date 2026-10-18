# Local Databases

SQLite database files are not distributed with this repository. Obtain the Superhero and Financial databases from the BIRD benchmark and place them here:

```text
database/superhero.sqlite
database/financial.sqlite
```

Dataset creation, NL2SQL inference, and GRPO training require the corresponding database. The training runner also accepts `--db-dir /path/to/databases` when the files are stored elsewhere. The released JSON instances and saved evaluation categories can be inspected without downloading databases.

These files are excluded by `.gitignore`. Database-dependent offline tests skip when the files are absent.
