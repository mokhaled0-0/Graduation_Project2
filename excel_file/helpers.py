import openpyxl


def readex(request):
    excel_file = request.FILES.get("ex_file")
    if excel_file == None:
        return ["NONE"]
    wb = openpyxl.load_workbook(excel_file)

    sheetsname = wb.sheetnames
    sheets = [wb[sheet] for sheet in sheetsname]
    sheet0 = sheets[0]
    text = []
    for i in range(2, 1000):
        t = sheet0['B' + str(i)].value
        if t == None:
            break
        text.append(t)
    return text
