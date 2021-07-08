
def generate_query(text, image_path):
    image_path = "/root/cogview2/text_samples/edge_samples/" + image_path
    print(f"[ROI1] {text} [BASE] [BOI1] [Image]{image_path} [EOI1] [ROI2] [BASE] [BOI2] [MASK]*1024 [EOI2]")
    pass

def generate_from_txt(txt_path):
    import json
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            text, image_path = line.split(' ')
            image_path = image_path.rstrip()
            generate_query(text, image_path)

def generate_test():
    text = "一只猪"
    image_path = "kafei.jpg"
    generate_query(text, image_path)
    pass

if __name__ == "__main__":
    generate_from_txt("querys/query.txt")
    pass