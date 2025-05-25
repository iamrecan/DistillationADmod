import os
import sys
import uuid
import zipfile
import time
import requests
import json
from dotenv import load_dotenv

# .env dosyasından çevre değişkenlerini yükle
load_dotenv()

nvai_url="https://ai.api.nvidia.com/v1/cv/meta/sam2"
nvai_polling_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
header_auth = f"Bearer {os.getenv('NVIDIA_API_KEY')}"

UPLOAD_ASSET_TIMEOUT = 300 # Timeout (in secs) to upload asset
MAX_RETRIES = 50 # Max num of retries while polling
DELAY_BTW_RETRIES = 2 # adding 1s delay between each polls


def _upload_asset(input, description):
    assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"

    headers = {
        "Authorization": header_auth,
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    s3_headers = {
        "x-amz-meta-nvcf-asset-description": description,
        "content-type": "video/mp4",
    }

    payload = {"contentType": "video/mp4", "description": description}

    response = requests.post(assets_url, headers=headers, json=payload, timeout=30)

    response.raise_for_status()

    asset_url = response.json()["uploadUrl"]
    asset_id = response.json()["assetId"]

    response = requests.put(
        asset_url,
        data=input,
        headers=s3_headers,
        timeout=UPLOAD_ASSET_TIMEOUT,
    )

    response.raise_for_status()
    return uuid.UUID(asset_id)


if __name__ == "__main__":
    """Uploads a video or image of your choosing to the NVCF API and sends a
    request to the SAM2 model. The response is saved to a
    local directory.

    Note: You must set up an environment variable, NVIDIA_API_KEY.
    """

    if len(sys.argv) != 4:
        print("Kullanım: python send_requestSam2.py <prompt> <input_video> <output_dir>")
        print("Prompt, aşağıdaki örneğe göre nokta istemlerinin JSON dizesi olmalıdır:")
        sample_prompt_json = ''' { "prompts": [
           {
             "type": "points",
             "object_id": 1,
             "frame_index": 0,
             "points": [
               {"x": 20, "y": 375, "label": true},
               {"x": 54, "y": 362, "label": false}
             ]
           },
           {
             "type": "points",
             "object_id": 2,
             "frame_index": 0,
             "points": [
                 {"x": 104, "y": 381,"label": true},
                 {"x": 109, "y": 437, "label": true},
                 {"x": 77, "y": 377, "label": false}]
           }
           ] } '''
        print(f"{sample_prompt_json}")
        sys.exit(1)

    nvidia_api_key = os.getenv('NVIDIA_API_KEY')
    if not nvidia_api_key or len(nvidia_api_key) == 0:
        print("Lütfen .env dosyanızda NVIDIA_API_KEY ortam değişkenini ayarlayın.")
        print("API anahtarınızı buradan alabilirsiniz: https://build.nvidia.com/")
        sys.exit(1)

    print(f"Kullanılan NVIDIA API Anahtarı: {nvidia_api_key[:10]}...")

    try:
        asset_id = _upload_asset(open(sys.argv[2], "rb"), "Input Video")
        print(f"Varlık başarıyla yüklendi, ID: {asset_id}")

        point_prompts = json.loads(f"{sys.argv[1]}")["prompts"]

        inputs = { 
            "model": "meta/sam2-hiera-large",
            "messages": [
              {
                "role": "user",
                "content": [
                    {
                    "type": "media_url",
                    "media_url": {
                      "url": f"data:video/mp4;asset_id,{asset_id}"
                    }
                 }
                ]
              }
            ],
            "add_objects": False
          }

        for point_prompt in point_prompts:
            inputs["messages"][0]["content"].append(point_prompt)

        asset_list = f"{asset_id}"

        headers = {
            "Content-Type": "application/json",
            "NVCF-INPUT-ASSET-REFERENCES": asset_list,
            "NVCF-FUNCTION-ASSET-IDS": asset_list,
            "Authorization": header_auth,
        }

        print(f"Girdi mesajı: {inputs}")
        print("NVIDIA API'sine istek gönderiliyor, lütfen yanıt için bekleyin...")
        response = requests.post(nvai_url, headers=headers, json=json.loads(json.dumps(inputs)))
        print(f"Yanıt durum kodu: {response.status_code}")

        if response.status_code != 200 and response.status_code != 202 and response.status_code != 302:
            print(f"Hata: Beklenmeyen durum kodu {response.status_code}")
            print(f"Yanıt: {response.text}")
            sys.exit(1)

        if response.status_code == 200: # değerlendirme tamamlandı, çıktı videosu hazır
            print("Değerlendirme tamamlandı! Sonuçlar kaydediliyor...")
            with open(f"{sys.argv[3]}.zip", "wb") as out:
                out.write(response.content)
            with zipfile.ZipFile(f"{sys.argv[3]}.zip", "r") as z:
                z.extractall(sys.argv[3])

        elif response.status_code == 202: # değerlendirme bekleniyor
            print("Değerlendirme bekleniyor ...")
            nvcf_reqid = response.headers['NVCF-REQID']
            nvai_polling_url = nvai_polling_url + nvcf_reqid

            # Yanıtın hazır olup olmadığını kontrol etmek için anket
            retries_left = MAX_RETRIES
            while retries_left > 0:
                print(f'Anket yapılıyor... ({MAX_RETRIES - retries_left + 1}/{MAX_RETRIES})')
                headers_polling = { "accept": "application/json", "Authorization": header_auth }
                response_polling = requests.get(nvai_polling_url, headers=headers_polling)
                
                if response_polling.status_code == 202: # değerlendirme bekleniyor
                    print('Sonuç henüz hazır değil.')
                    retries_left -= 1
                    time.sleep(DELAY_BTW_RETRIES)
                    continue
                elif response_polling.status_code == 200: # değerlendirme tamamlandı, çıktı videosu hazır
                    print('Sonuç hazır!')
                    with open(f"{sys.argv[3]}.zip", "wb") as out:
                        out.write(response_polling.content)
                    break
                else:
                    print(f"Beklenmeyen yanıt durumu: {response_polling.status_code}")
                    print(f"Yanıt: {response_polling.text}")
                    break

            if retries_left == 0:
                print("Maksimum deneme sayısına ulaşıldı. İstek hala işleniyor olabilir.")
                sys.exit(1)

            with zipfile.ZipFile(f"{sys.argv[3]}.zip", "r") as z:
                z.extractall(sys.argv[3])

        print(f"Çıktı {sys.argv[3]} konumuna kaydedildi")
        print("Çıktı dizinindeki dosyalar:")
        print(os.listdir(sys.argv[3]))

    except FileNotFoundError:
        print(f"Hata: Girdi dosyası bulunamadı: {sys.argv[2]}")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Hata: İstemci parametresinde geçersiz JSON formatı")
        sys.exit(1)
    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        sys.exit(1)