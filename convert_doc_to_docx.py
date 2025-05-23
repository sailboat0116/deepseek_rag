import os
import win32com.client as win32

folder = r"C:\Users\sailboat\data"

word = win32.gencache.EnsureDispatch('Word.Application')
for file in os.listdir(folder):
    if file.endswith(".doc") and not file.endswith(".docx"):
        doc_path = os.path.join(folder, file)
        docx_path = os.path.join(folder, file + "x")  # .docx
        print(f"🔄 轉換中：{file}")
        doc = word.Documents.Open(doc_path)
        doc.SaveAs(docx_path, FileFormat=16)  # 16 = wdFormatDocumentDefault
        doc.Close()
word.Quit()
print("✅ 所有 .doc 檔已轉換為 .docx")
