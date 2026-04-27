# Budapest Airbnb Price Estimator 🇭🇺🏡

Egy modern, interaktív webalapú dashboard és gépi tanulásos árbecslő alkalmazás, amely a budapesti Airbnb piac adatait elemzi, és segít a szállásadóknak az optimális éjszakánkénti ár meghatározásában.

## 🌟 Funkciók
* **Piaci Térkép & Hőtérkép**: A budapesti szállások átlagárainak és eloszlásának vizualizációja interaktív Plotly térképeken (sűrűség és kerületi buborék nézet).
* **Statisztikák & Elemzések**: Top 10 legdrágább kerület, árak eloszlása szobatípusonként, és általános piaci mutatók (medián ár, átlagos értékelések).
* **Árbecslő Modell**: Egy `HistGradientBoostingRegressor` alapú gépi tanulási modell, amely a lakás elhelyezkedése (kerület), mérete (hálók, fürdők száma), típusa, és a várható értékelések alapján megbecsüli az optimális kiadási árat.

## 🛠 Technológiai Stack
* **Nyelv**: Python 3.x
* **Webes keretrendszer**: [Streamlit](https://streamlit.io/) (egyedi, sötét dizájnnal és üveghatással - *glassmorphism*)
* **Adatfeldolgozás**: Pandas, NumPy
* **Gépi Tanulás**: Scikit-learn (HistGradientBoostingRegressor, Pipeline, RandomizedSearchCV), Joblib
* **Vizualizáció**: Plotly Express & Graph Objects

## 🚀 Futtatás lokálisan

### 1. Repository klónozása
```bash
git clone https://github.com/so28u8/airbnb-price-guesser.git
cd airbnb-price-guesser
```

### 2. Függőségek telepítése
Hozd létre és aktiváld a virtuális környezetet, majd telepítsd a szükséges csomagokat:
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install streamlit pandas numpy scikit-learn plotly joblib
```

### 3. Adatkészlet letöltése (opcionális a futtatáshoz, kötelező a tanításhoz)
Mivel az eredeti `listings.csv` fájl túl nagy, nem része a repozitóriumnak. Ha újra szeretnéd tanítani a modellt:
1. Töltsd le a budapesti adatokat az [Inside Airbnb](https://insideairbnb.com/get-the-data/) oldalról.
2. Nevezd el `listings.csv`-nek és tedd a projekt gyökérmappájába.
3. Futtasd a modell tanítását: `python program/model_trainer.py`

*(A repó tartalmazza a már betanított modellt a `program/model/` mappában, így anélkül is elindul az alkalmazás!)*

### 4. Az alkalmazás indítása
A projekt gyökérmappájából (vagy a `program` mappából) indítsd el a Streamlit szervert:
```bash
streamlit run program/app.py
```
Az alkalmazás automatikusan megnyílik a böngésződben a `http://localhost:8501` címen.

## 📁 Projekt struktúra
* `program/app.py` - A Streamlit dashboard és a UI fő kódja.
* `program/data_loader.py` - Az adatok tisztításáért és betöltéséért felelős szkript.
* `program/model_trainer.py` - A gépi tanulási modell betanítását és optimalizálását végző szkript.
* `program/model/` - A már betanított modellt (`price_model.pkl`) és kerületi statisztikákat tartalmazó mappa.
* `spec.pdf` - A projekt eredeti specifikációja.
