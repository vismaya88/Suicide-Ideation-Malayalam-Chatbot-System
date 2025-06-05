import evaluate
from sklearn.metrics import accuracy_score
from bert_score import score
from tabulate import tabulate

# Sample data
inputs = [
    "എന്റെ നെഞ്ച് വേദനിക്കുന്നു. എനിക്ക് മരണം ആഗ്രഹമുണ്ട്.",
    "എനിക്ക് ഉത്കണ്ഠ ഉണ്ടാകുന്നുണ്ട്. ഉറക്കമില്ല."
]

true_severities = ["Suicidal", "Anxiety"]
predicted_severities = ["Suicidal", "Anxiety"]

true_responses = [
    "നമസ്കാരം, നിങ്ങൾ അനുഭവിക്കുന്ന വേദന വളരെ ഗുരുതരമാണ്. ദയവായി അടിയന്തരമായി ഒരു കൗൺസിലറെ സമീപിക്കുക. നിങ്ങൾക്ക് പിന്തുണ ലഭ്യമാകുന്നതാണ്.",
    "നമസ്കാരം, ഉത്കണ്ഠം സാധാരണമാണ്. ആത്മവിശ്വാസം നിലനിർത്തുക. ആവശ്യമായാൽ വിദഗ്ധന്റെ സഹായം തേടുക."
]

generated_responses = [
    "നമസ്കാരം, നിങ്ങളുടെ അനുഭവം അതിയായ വേദനാജനകമാണെന്ന് ഞാൻ മനസിലാക്കുന്നു. നിങ്ങൾക്ക് സഹായം ആവശ്യമാണ്. ദയവായി ഒരു മനശ്ശാസ്ത്രജ്ഞനെയോ നിങ്ങളുടെ വിശ്വസനീയരായ ആരെയെങ്കിലും സമീപിക്കുക. നിങ്ങൾക്ക് ജീവിതത്തിൽ ഇന്നും പ്രതീക്ഷയുണ്ട്.",
    "നമസ്കാരം, നിങ്ങൾ നേരിടുന്ന ഉത്കണ്ഠ സാധാരണമാണ്. നിങ്ങളുടെ സ്വഭാവം മനസ്സിലാക്കുന്നത് പ്രാധാന്യമുള്ള ഒരു പടിയാകാം. സുഖം ലഭിക്കാൻ ഒരു ആശ്വാസ സംഭാഷണം കൂടെ സഹായകരമായിരിക്കും. ആവശ്യമായെങ്കിൽ ചികിത്സ തേടുക."
]



# Severity prediction accuracy
severity_acc = accuracy_score(true_severities, predicted_severities)

# BERTScore
P, R, F1 = score(generated_responses, true_responses, lang="ml", verbose=True)

# Display Results
print("\nAspect-wise Evaluation")
print(tabulate([
    ["Severity Detection Accuracy", f"{severity_acc * 100:.2f} %"],
    ["Avg BERTScore F1", f"{F1.mean().item():.4f}"]
], headers=["Aspect", "Metric Value"]))
