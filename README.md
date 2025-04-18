# Nur AI - Advanced Transformer Model

Nur AI, modüler ve genişletilebilir bir transformer tabanlı AI modelidir.

## Proje Yapısı

```
nur_ai/
│
├── configs/
│   └── nur_base_config.json
│
├── experiments/
│   └── (deney sonuçları bu dizinde olacak)
│
├── logs/
│   └── (log dosyaları bu dizinde olacak)
│
├── models/
│   └── (eğitilmiş modeller bu dizinde olacak)
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── training.py
│   └── utils.py
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_config.py
│   ├── test_model.py
│   └── test_utils.py
│
├── scripts/
│   ├── run_tests.py
│   └── train_model.py
│
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── pyproject.toml
├── README.md
├── requirements.txt
└── setup.py
```

## Kurulum

Proje bağımlılıklarını yüklemek için:

```bash
pip install -r requirements.txt
```

## Kullanım

Modeli eğitmek için:

```bash
python scripts/train_model.py
```

Testleri çalıştırmak için:

```bash
python scripts/run_tests.py
```

Docker kullanarak çalıştırmak için:

```bash
docker-compose up --build
```
