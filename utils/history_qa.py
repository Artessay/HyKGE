def history_qa_process(qa_epoch, data, history, history_epoch):
    if qa_epoch == 0:
        history = []
    if len(history) == history_epoch:
        history.pop(0)
    history.append(data)

    cat_data = ''
    for data_item in history:
        cat_data = cat_data + ',' + data_item
    print(cat_data)
    return cat_data, history