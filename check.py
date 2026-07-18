import fitz  # PyMuPDF

path="/media/parmpal/Data1/Books/some-investment-books-master/Forex Trading Secrets - Trading Strategies for the Forex Market 2010.pdf"
doc = fitz.open(path)

print(doc)
print(doc.get_toc(simple=True)  )