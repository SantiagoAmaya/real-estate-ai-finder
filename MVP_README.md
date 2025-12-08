# 🏠 Real Estate AI Finder - MVP

**Find properties that match complex requirements using AI**

A production-ready web application that uses multi-modal AI (text + vision) to analyze real estate listings and rank them based on natural language queries in Spanish.

---

## 🎯 What It Does

**Instead of this** (traditional search):
```
Location: Barcelona
Price: 100k-300k
Rooms: 2+
Type: Local
```

**Use natural language**:
```
"Local con entrada independiente en Barcelona,
 luminoso, techos altos, máximo 250mil euros"
```

The AI understands complex requirements that traditional filters can't handle:
- ✅ "entrada independiente" (independent entrance)
- ✅ "luminoso" (bright/natural light)
- ✅ "techos altos" (high ceilings)
- ✅ "reformado" (renovated)
- ✅ And many more...

---

## 🏗️ Architecture

```
┌──────────────┐
│   FRONTEND   │  Streamlit (Python)
│  (User UI)   │  - API key input
└──────┬───────┘  - Search interface
       │          - Results display
       │
       │ HTTPS
       │
┌──────▼───────┐
│   BACKEND    │  FastAPI (Python)
│    (API)     │  - Query parsing
└──────┬───────┘  - Property scraping
       │          - AI analysis
       │          - Ranking
       │
       │
┌──────▼───────┐
│              │  - CombinedPropertyAnalyzer
│  (ML Logic)  │  - Text analysis (API/Local)
└──────────────┘  - Vision analysis (Claude/Qwen)
```

---

## 🚀 Quick Start

### Local Testing (5 minutes)

```bash
# 1. Start backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# 2. Start frontend (new terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py

# 3. Open browser: http://localhost:8501
# 4. Enter your Anthropic API key
# 5. Search: "Local entrada independiente Barcelona"
```

See [LOCAL_TESTING_GUIDE.md](LOCAL_TESTING_GUIDE.md) for details.

### Production Deployment (1-2 hours)

1. **Backend → Railway** (€5-10/month)
2. **Frontend → Streamlit Cloud** (FREE)

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for step-by-step instructions.

---

## 📂 Project Structure

```
real-estate-ai-finder/
├── backend/                    # FastAPI backend
│   ├── main.py                # API endpoints
│   ├── Dockerfile             # Container config
│   └── requirements.txt       # Dependencies
│
├── frontend/                   # Streamlit frontend
│   ├── app.py                 # UI application
│   ├── Dockerfile             # Container config
│   ├── requirements.txt       # Dependencies
│   └── .streamlit/
│       ├── config.toml        # Streamlit config
│       └── secrets.toml       # Backend URL (not committed)
│
├── src/                        # Your ML code
│   ├── query_parser/          # NLP query parsing
│   ├── data/                  # Scraping
│   ├── property_analysis/     # Text + Vision analysis
│   └── ...
│
├── scripts/                    # Testing scripts
├── data/                       # Data storage
└── docs/                       # Documentation
```

---

## 🔑 Features

### For Users

- 🔍 **Natural language search** in Spanish
- 🤖 **AI-powered analysis** of text + images
- 📊 **Ranked results** with match scores
- 💰 **Cost tracking** per search
- 🔐 **Secure** - use your own API key

### For Developers

- ⚡ **Fast deployment** (1-2 hours to production)
- 💵 **Low cost** (€5-10/month infrastructure)
- 🐳 **Dockerized** (easy to deploy anywhere)
- 📝 **Auto-documented** API (FastAPI Swagger)
- 🧪 **Easy testing** (local + production)

---

## 💰 Cost Breakdown

### Infrastructure

| Service | Cost | What It Does |
|---------|------|--------------|
| Streamlit Cloud | **FREE** | Hosts frontend |
| Railway | **€5-10/month** | Hosts backend API |

### Per Search (User pays with their API key)

| Mode | Text | Vision | Total |
|------|------|--------|-------|
| API + Claude | €0.003 | €0.075 | €0.078 |
| API + Qwen | €0.003 | €0 | €0.003 ⭐ |
| Local + Qwen | €0 | €0 | €0 |

**Recommended:** API + Qwen = €0.003 per search

**Example:** 100 searches/month = €0.30 user cost + €5 infra = **€5.30 total**

---

## 🛠️ Technology Stack

### Frontend
- **Streamlit** - Python web framework
- **Requests** - HTTP client
- **Pandas** - Data display

### Backend
- **FastAPI** - Modern Python API framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### ML/AI (Your existing code)
- **Claude API** - Text analysis + Vision
- **Sentence-transformers** - Local embeddings
- **Qwen2-VL** - Local vision (optional)
- **BeautifulSoup** - Web scraping

### Deploy
- **Railway** - Backend hosting
- **Streamlit Cloud** - Frontend hosting
- **Docker** - Containerization

---

## 📖 Documentation

- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Deploy to Railway + Streamlit Cloud
- **[LOCAL_TESTING_GUIDE.md](LOCAL_TESTING_GUIDE.md)** - Test locally before deploy
- **[LOCAL_GPU_IMPLEMENTATION.md](LOCAL_GPU_IMPLEMENTATION.md)** - Use local GPU instead of API

---

## 🔒 Security

### API Key Handling
- ✅ Users provide their own API key
- ✅ Never stored in database
- ✅ Only in browser session memory
- ✅ Validated before use

### Best Practices
- 🚫 Never commit API keys to Git
- ✅ Use secrets management (Streamlit secrets, Railway env vars)
- ✅ CORS properly configured
- ✅ Input validation on all endpoints

---

## 🧪 Testing

### Quick Test (Locally)
```bash
# Terminal 1: Backend
cd backend && uvicorn main:app

# Terminal 2: Frontend  
cd frontend && streamlit run app.py

# Terminal 3: Test
curl http://localhost:8000/health
# Should return: {"status":"healthy",...}
```

### Full Test (Production)
```bash
# After deploying to Railway + Streamlit Cloud

# 1. Check backend
curl https://your-app.railway.app/health

# 2. Check frontend
open https://your-app.streamlit.app

# 3. Test search E2E
# Enter API key → Search → See results
```

See [LOCAL_TESTING_GUIDE.md](LOCAL_TESTING_GUIDE.md) for comprehensive tests.

---

## 🚀 Deployment Steps (Summary)

1. **Prepare code:**
```bash
git add backend/ frontend/
git commit -m "feat: Add MVP"
git push
```

2. **Deploy backend to Railway:**
   - Go to railway.app
   - New Project → From GitHub
   - Select repo → Auto-deploy
   - Get URL: `https://your-app.railway.app`

3. **Deploy frontend to Streamlit Cloud:**
   - Go to share.streamlit.io
   - New app → Select repo
   - Main file: `frontend/app.py`
   - Add secret: `BACKEND_URL = "https://your-app.railway.app"`
   - Deploy → Get URL: `https://your-app.streamlit.app`

4. **Test:**
   - Open Streamlit URL
   - Enter API key
   - Search → Results!

**Total time:** 1-2 hours first time, 5 minutes after practice.

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

---

## 📊 Performance

### Expected Response Times

| Operation | Time |
|-----------|------|
| Parse query | 1s |
| Scrape properties (2 pages) | 10-15s |
| Text analysis (10 props) | 5-10s |
| Vision analysis (5 props) | 20-30s |
| **Total (new scrape)** | **35-55s** |
| **Total (cached)** | **10-20s** |

### Optimization Tips
- ✅ Use cached data for testing (much faster)
- ✅ Limit max_results to 10 (default)
- ✅ Use vision agent (analyzes only top candidates)
- ✅ Consider qwen_only mode (free vision)

---

## 🛣️ Roadmap

### ✅ Phase 3: Analysis Models (COMPLETE)
- Text analysis (API + Local)
- Vision analysis (Claude + Qwen)
- Multi-modal scoring

### ✅ Phase 4: MVP (CURRENT)
- FastAPI backend
- Streamlit frontend
- Docker containers
- Production deployment

### 🔜 Phase 5: Monitoring (NEXT)
- Usage analytics
- Cost tracking
- Performance monitoring
- Error alerting

### 🔜 Phase 6: Polish
- Custom domain
- User authentication (optional)
- Search history with DB
- Advanced filters UI
- Mobile responsive

---

## 🤝 Contributing

This is a PhD research project, but suggestions welcome!

**Found a bug?**
- Check logs in Railway dashboard
- Check browser console (F12)
- Open an issue with details

**Have a feature idea?**
- Open an issue describing the use case
- Consider if it fits the MVP scope

---

## 📄 License

[Your License Here]

---

## 👤 Author

Santiago Amaya
- PhD Student in Multi-Agent RL & MLOps
- Project: Intelligent Real Estate Search System

---

## 🙏 Acknowledgments

- **Anthropic** - Claude API for text analysis
- **Streamlit** - Amazing Python web framework
- **Railway** - Simple deployment platform
- **Fotocasa** - Property data source

---

## 📞 Support

**Documentation:**
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - How to deploy
- [LOCAL_TESTING_GUIDE.md](LOCAL_TESTING_GUIDE.md) - How to test
- [LOCAL_GPU_IMPLEMENTATION.md](LOCAL_GPU_IMPLEMENTATION.md) - GPU setup

**External:**
- Railway Docs: https://docs.railway.app/
- Streamlit Docs: https://docs.streamlit.io/
- FastAPI Docs: https://fastapi.tiangolo.com/

---

## ⭐ Star This Repo

If you find this project useful, please give it a star! ⭐

It helps others discover the project and motivates continued development.

---

**Built with ❤️ for the real estate search problem**