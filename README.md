<div align="center">

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/TMDB_API-01D277?style=for-the-badge&logo=themoviedatabase&logoColor=white"/>
<img src="https://img.shields.io/badge/Deployed-Live-brightgreen?style=for-the-badge&logo=streamlit"/>

<br/><br/>

```
███╗   ██╗███████╗████████╗███████╗██╗     ██╗██╗  ██╗    ██████╗ ██╗
████╗  ██║██╔════╝╚══██╔══╝██╔════╝██║     ██║╚██╗██╔╝    ██╔══██╗██║
██╔██╗ ██║█████╗     ██║   █████╗  ██║     ██║ ╚███╔╝     ███████║██║
██║╚██╗██║██╔══╝     ██║   ██╔══╝  ██║     ██║ ██╔██╗     ██╔══██║██║
██║ ╚████║███████╗   ██║   ██║     ███████╗██║██╔╝ ██╗    ██║  ██║██║
╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝     ╚══════╝╚═╝╚═╝  ╚═╝   ╚═╝  ╚═╝╚═╝
```

# 🎬 Netflix AI Movie Recommender

### *Discover your next favorite film — powered by Machine Learning*

<br/>

[![🚀 Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Click%20Here-E50914?style=for-the-badge)](https://netflix-ai-movie-recommender-zwnhjvhqog678y7bjpphbe.streamlit.app)
&nbsp;
[![⭐ Star this Repo](https://img.shields.io/github/stars/yourusername/netflix-ai-movie-recommender?style=for-the-badge&color=FFD700)](https://github.com/yourusername/netflix-ai-movie-recommender)
&nbsp;
[![🍴 Fork](https://img.shields.io/github/forks/yourusername/netflix-ai-movie-recommender?style=for-the-badge&color=6C63FF)](https://github.com/yourusername/netflix-ai-movie-recommender/fork)

<br/>

> *"Pick any movie. Let AI do the rest."*

<br/>

---

</div>

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Live Demo](#-live-demo)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Machine Learning Architecture](#-machine-learning-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Screenshots](#-screenshots)
- [Dataset](#-dataset)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)

---

## 🎯 About the Project

**Netflix AI Movie Recommender** is a full-stack, production-deployed machine learning application that recommends movies based on content similarity. Built with a sleek **Netflix-inspired dark UI**, this project combines the power of **Natural Language Processing**, **K-Nearest Neighbors**, and **Cosine Similarity** to deliver personalized, intelligent movie suggestions — complete with real movie posters, ratings, release dates, and descriptions fetched live from the **TMDB API**.

This is not just a recommendation engine — it's a fully deployed, interactive web application accessible to anyone, anywhere, right now.

---

## 🌐 Live Demo

<div align="center">

### 👉 [https://netflix-ai-movie-recommender-zwnhjvhqog678y7bjpphbe.streamlit.app](https://netflix-ai-movie-recommender-zwnhjvhqog678y7bjpphbe.streamlit.app)

*No installation required. Open in your browser and start discovering movies instantly.*

</div>

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **AI-Powered Recommendations** | Content-based ML engine suggests the 5 most similar movies |
| 🎨 **Netflix-Style Dark UI** | Immersive, responsive, cinema-grade interface |
| 🖼️ **Live Movie Posters** | Real-time poster fetching via TMDB API |
| ⭐ **Ratings & Release Dates** | Full movie metadata displayed per recommendation |
| 📝 **Movie Overviews** | Plot summaries for each recommended title |
| 📱 **Responsive Card Layout** | Clean grid layout that works on all screen sizes |
| ⚡ **Real-Time Results** | Instant recommendations on movie selection |
| ☁️ **Cloud Deployed** | Live on Streamlit Cloud — no setup needed |

---

## 🧠 How It Works

```
User Selects a Movie
        │
        ▼
┌───────────────────────────────────────────────┐
│         Feature Extraction Pipeline           │
│                                               │
│  Title + Genre + Keywords + Cast + Crew       │
│              │                                │
│              ▼                                │
│      Text Preprocessing (NLP)                 │
│   (Lowercasing, Stemming, Tokenization)        │
│              │                                │
│              ▼                                │
│     CountVectorizer (Bag of Words)            │
│    → 5000-dimensional feature vector          │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│         Similarity Computation               │
│                                               │
│    Cosine Similarity Matrix (NxN)             │
│    KNN identifies K closest neighbors        │
│                                               │
│  similarity = cos(θ) = (A·B) / (‖A‖ × ‖B‖)   │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│         TMDB API Enrichment                  │
│                                               │
│  Fetches: Poster · Rating · Date · Overview   │
└───────────────────────────────────────────────┘
        │
        ▼
   Top 5 Recommendations Displayed
   in Netflix-Style Card Layout 🎬
```

---

## 🔬 Machine Learning Architecture

### Algorithm: Content-Based Filtering

This system recommends movies by analyzing **what a movie is about**, not who watched it. It uses a combination of:

#### 1. 📊 Feature Engineering
Movies are described by a **combined metadata tag** built from:
- Movie genres
- Plot keywords
- Top cast members
- Director name
- Movie overview

#### 2. 🔤 Text Vectorization
The combined tags are transformed into numerical vectors using `CountVectorizer` with a vocabulary of **5,000 features**, converting each movie into a point in high-dimensional space.

#### 3. 📐 Cosine Similarity
The angle between two movie vectors determines their similarity. Movies with smaller angles (higher cosine scores) are more alike — regardless of their magnitude.

```
Cosine Similarity = (A · B) / (||A|| × ||B||)  →  Range: [0, 1]
```

#### 4. 🔍 K-Nearest Neighbors (KNN)
KNN efficiently finds the **K most similar movies** in the feature space without requiring model retraining, making it ideal for recommendation tasks.

| Metric | Value |
|---|---|
| Algorithm | Content-Based Filtering |
| Vectorizer | CountVectorizer |
| Vocabulary Size | 5,000 features |
| Similarity Metric | Cosine Similarity |
| Neighbor Search | K-Nearest Neighbors |
| Recommendations | Top 5 similar movies |

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.9+ | Core backend logic |
| **Web Framework** | Streamlit | Interactive web application |
| **Data Processing** | Pandas, NumPy | Dataset manipulation |
| **Machine Learning** | Scikit-learn | KNN + Cosine Similarity |
| **NLP** | NLTK (Stemming) | Text preprocessing |
| **Movie Data** | TMDB API | Posters, ratings, metadata |
| **Serialization** | Pickle | Model & data persistence |
| **Version Control** | Git & GitHub | Source control |
| **Deployment** | Streamlit Cloud | Live hosting |

</div>

---

## 📂 Project Structure

```
netflix-ai-movie-recommender/
│
├── 📁 dataset/
│   ├── tmdb_5000_movies.csv       # Movie metadata (5000 films)
│   └── tmdb_5000_credits.csv      # Cast & crew information
│
├── 📄 main.py                     # Data preprocessing & model training
├── 📄 recommender.py              # Core recommendation engine (ML logic)
├── 📄 app.py                      # Streamlit UI + TMDB API integration
├── 📄 requirements.txt            # Project dependencies
└── 📄 README.md                   # Project documentation
```

### File Responsibilities

| File | Role |
|---|---|
| `main.py` | Loads the TMDB dataset, engineers features, builds the similarity matrix, and serializes the model using Pickle |
| `recommender.py` | Contains the KNN + Cosine Similarity logic; given a movie title, returns the top-N most similar movies |
| `app.py` | Streamlit frontend; handles user input, calls `recommender.py`, fetches live data from TMDB API, and renders the Netflix-style UI |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- A free [TMDB API Key](https://www.themoviedb.org/settings/api)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/netflix-ai-movie-recommender.git
cd netflix-ai-movie-recommender
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your TMDB API Key

Open `app.py` and replace the placeholder:

```python
API_KEY = "your_tmdb_api_key_here"
```

### 4. Run the ML Pipeline (First-Time Setup)

```bash
python main.py
```

> This processes the dataset and generates the similarity matrix. Only needs to run once.

### 5. Launch the Application

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** and start exploring!

---

## 📦 Requirements

```txt
streamlit
pandas
numpy
scikit-learn
nltk
requests
pickle-mixin
```

---

## 🗂️ Dataset

This project uses the **TMDB 5000 Movie Dataset** from Kaggle:

| File | Records | Key Columns |
|---|---|---|
| `tmdb_5000_movies.csv` | 4,803 movies | title, genres, keywords, overview, vote\_average |
| `tmdb_5000_credits.csv` | 4,803 entries | cast, crew |

📥 [Download from Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## 🗺️ Future Roadmap

- [ ] 🔐 User authentication & personalized watchlists
- [ ] 👥 Collaborative filtering for user-based recommendations
- [ ] 🎭 Genre & mood-based filtering
- [ ] 🔎 Full-text movie search with fuzzy matching
- [ ] 📊 Recommendation confidence scores
- [ ] 🌍 Multi-language support
- [ ] 🎞️ Trailer embedding via YouTube API
- [ ] 🧪 A/B testing for recommendation algorithms

---

## 🤝 Contributing

Contributions are what make the open-source community amazing. Any contributions you make are **greatly appreciated**.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 Author

<div align="center">

**Built with ❤️ and lots of ☕**

*If this project helped you or impressed you, please consider giving it a ⭐ — it means the world!*

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/yourusername)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/yourusername)

<br/>

---

*"The best way to predict the future is to build it."*

</div>
