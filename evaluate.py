from chatbot import rechercher

# (question, mots-clés attendus dans le nom ou la catégorie)
TESTS = [
    ("wireless bluetooth headphones", ["headphone", "headset", "earphone"]),
    ("wrist watch for men under 1500", ["watch"]),
    ("women cycling shorts", ["short"]),
    ("gold necklace", ["necklace", "pendant", "chain"]),
    ("mobile back cover case", ["case", "cover"]),
    ("power bank", ["power bank", "powerbank"]),
    ("usb cable", ["cable"]),
    ("men casual shoes", ["shoe", "sneaker", "loafer"]),
    ("ball pen", ["pen"]),
    ("kitchen storage container", ["container", "jar", "box"]),
]

total, ok_total = 0, 0
for question, mots in TESTS:
    docs = rechercher(question)
    n_ok = sum(
        any(m in (d.metadata["product_name"] + " " + d.metadata["categorie"]).lower()
            for m in mots)
        for d in docs
    )
    total += len(docs)
    ok_total += n_ok
    print(f"{n_ok}/{len(docs)}  {question}")

if total:
    print(f"\nPrécision@5 moyenne : {ok_total / total:.0%}")