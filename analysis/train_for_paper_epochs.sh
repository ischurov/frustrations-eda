
PYTHON=python

mkdir paper_final_epochs

${PYTHON} train_for_paper_epochs.py square truncated
${PYTHON} train_for_paper_epochs.py square flat
${PYTHON} train_for_paper_epochs.py square nosign
 
