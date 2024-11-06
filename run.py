import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

img = cv2.imread('D:/aa.jpg')

if img is None:
    print("Görüntü bulunamadı veya okunamıyor. Dosya yolunu kontrol et.")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            if 2 < aspect_ratio < 5:
                plate_img = img[y:y + h, x:x + w]
                plate_img_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                plate_img_binary = cv2.threshold(plate_img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                plate_text = pytesseract.image_to_string(plate_img_binary, config='--psm 8')

                print("Plaka Tanındı:", plate_text.strip())

                cv2.imshow('Detected Plate', plate_img)
                break

    cv2.imshow('Original Image with Plate Contour', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
