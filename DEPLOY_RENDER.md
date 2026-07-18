# Hosting στο Render με SQLite

## Γιατί χρειάζεται persistent disk

Η SQLite αποθηκεύει όλα τα δεδομένα σε αρχείο. Το συνηθισμένο filesystem ενός Render Web Service είναι προσωρινό, επομένως οι αλλαγές μπορούν να χαθούν μετά από restart ή redeploy.

Για μόνιμη αποθήκευση χρησιμοποίησε:

- paid Render Web Service
- persistent disk
- mount path: `/var/data`
- environment variable: `DATABASE_PATH=/var/data/llm_eval.db`

Το repository περιλαμβάνει ήδη `render.yaml`, άρα ο ευκολότερος τρόπος είναι Blueprint deployment.

## 1. Ανέβασμα στο GitHub

Αντικατέστησε τα αρχεία του repository με τα αρχεία αυτού του φακέλου και εκτέλεσε:

```bash
git add .
git commit -m "Add corrected SQLite benchmark platform"
git push origin main
```

Το `.env` δεν πρέπει να ανέβει.

## 2. Δημιουργία Blueprint στο Render

1. Άνοιξε το Render Dashboard.
2. Επίλεξε **New → Blueprint**.
3. Σύνδεσε το repository `llm-evaluation-platform`.
4. Το Render θα διαβάσει το `render.yaml`.
5. Όρισε τιμή για το secret environment variable `APP_PASSWORD`.
6. Ολοκλήρωσε τη δημιουργία του service.

Το Blueprint ορίζει:

```text
Build command:
pip install -r requirements.txt

Start command:
python -m streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true

Database path:
/var/data/llm_eval.db

Disk:
1 GB mounted at /var/data
```

## 3. Πρώτη εκκίνηση

Κατά την πρώτη εκκίνηση, αν το `/var/data/llm_eval.db` δεν υπάρχει, η εφαρμογή αντιγράφει αυτόματα το:

```text
seed/llm_eval_seed.db
```

Έτσι το hosted dashboard εμφανίζει αμέσως το ολοκληρωμένο run `run_20260718_134525`.

## 4. Νέα runs

Τα νέα benchmarks είναι καλύτερο να εκτελούνται τοπικά, ώστε να ελέγχεις το κόστος και να μη μένει το OpenRouter API key στο hosted dashboard.

Μετά από νέο run, άνοιξε το hosted app και πήγαινε στο tab **Admin import**. Ανέβασε:

```text
run_<id>_metadata.json
run_<id>_results.json
run_<id>_pairwise_judgments.json
```

Η εφαρμογή θα:

1. διορθώσει ξανά τις deterministic μετρικές,
2. δημιουργήσει summaries και confidence intervals,
3. δημιουργήσει νέο stratified human sample,
4. αποθηκεύσει το run στην persistent SQLite βάση.

Δεν επαναλαμβάνονται API calls.

## 5. Blind human review

Στο tab **Blind human review** υπάρχουν 30 prompts:

- 3 ανά κατηγορία,
- 1 easy, 1 medium και 1 hard,
- 15 φορές Qwen ως Answer A,
- 15 φορές Gemma ως Answer A.

Οι αξιολογήσεις αποθηκεύονται στο SQLite αρχείο του persistent disk.

## Free Render

Μπορείς να ανεβάσεις το dashboard σε free Web Service μόνο για read-only παρουσίαση του υπάρχοντος seed benchmark. Χωρίς persistent disk, νέα imports και human-review εγγραφές μπορεί να χαθούν μετά από restart ή redeploy.
