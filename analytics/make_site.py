import webbrowser
import cv2
import os


def img_markup(out, name, image, height=128):
    cv2.imwrite(os.path.join(out, name), image)
    h = f'<img src="{name}" alt="{name}" style="height:{height}px;">'
    return h


def img_path_markup(name, height=128):
    h = f'<img src="{name}" alt="{name}" style="height:{height}px;">'
    return h


def make_website(out_folder, elements):
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    base = open('analytics/site_base.html', 'r')
    base = base.readlines()

    body_idx = base.index('$BODY$\n')

    website_path = f'{out_folder}/site_base.html'
    out = open(website_path, 'w')

    body = []
    img_idx = 0
    for element_type, element in elements:
        if element_type == "str":
            eh = [f"<p>{element}</p>"]
        elif element_type == "img":
            eh = [img_markup(out_folder, f"{img_idx}.jpg", element)]
            img_idx += 1
        elif element_type == "img_path":
            eh = [img_path_markup(element)]

        body += eh

    out_text = base[:body_idx] + body + base[body_idx+1:]

    out.writelines(out_text)
    out.close()

    webbrowser.open_new_tab(website_path)





