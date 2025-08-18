# BC AI Survey Data Storytelling: 3D Semantic Visualizations

> 1,001 British Columbians shared their thoughts on artificial intelligence's role in the province's future. This project transforms their raw responses into interactive 3D semantic visualizations that reveal patterns, concerns, and hopes.

> Using advanced text embeddings and machine learning, this data storytelling project creates immersive 3D scatter plots where similar responses naturally cluster together, allowing for intuitive exploration of public sentiment.

View the project here: **[Interactive Dashboard](index.html)**

## Pitch

Understanding public opinion on complex topics like artificial intelligence requires more than traditional surveys and charts. How do we comprehend the nuanced perspectives of over 1,000 British Columbians when they share their hopes, fears, and advice about AI?

This project applies semantic embedding technology—the same approach used in modern AI systems—to transform text responses into numerical representations that capture meaning. When visualized in 3D space using t-SNE dimensionality reduction, similar responses cluster together naturally, creating "semantic maps" of public opinion.

Imagine exploring a 3D landscape where each point represents a person's response, and similar thoughts gravitate toward each other. You can rotate, zoom, and discover authentic voices from across British Columbia, understanding not just what people think, but how their perspectives relate to one another.

These interactive visualizations make complex survey data accessible and reveal patterns that traditional analysis might miss.

## Approach

This project builds on proven methodologies for semantic visualization of survey data, adapted specifically for the BC AI Survey dataset. The technical approach involves:

1. **Text Embedding Generation**: Each open-ended survey response was converted into a 3,072-dimensional vector using OpenAI's `text-embedding-3-large` model. These embeddings capture semantic meaning, allowing mathematical comparison of conceptual similarity.

2. **Dimensionality Reduction**: The high-dimensional embeddings were reduced to 3D coordinates using **t-SNE** (t-distributed Stochastic Neighbor Embedding), preserving local clustering patterns while making visualization possible.

3. **Optimal Cluster Identification**: **K-Means clustering** was applied to the 3D coordinates with **Silhouette Score** optimization to programmatically determine the ideal number of thematic clusters for each question.

4. **Response Aggregation**: Identical text responses were aggregated into single points, with marker sizes calculated proportional to the **cube root of frequency**—ensuring point volume accurately represents the number of people sharing that perspective.

5. **Interactive 3D Visualization**: **Plotly** generates fully interactive 3D scatter plots where users can rotate, pan, zoom, and hover over points to read original responses, view cluster information, and understand response frequency.

## Dataset: BC AI Survey

This project analyzes survey data from **1,001 British Columbians** collected about artificial intelligence's role in the province. The dataset includes:

-   **16 open-ended text questions** covering AI experience, concerns, hopes, and advice
-   **Rich demographic information** including age, location, education, and income
-   **Geographic distribution**: Vancouver (76.6%), Victoria (13.7%), rural BC (9.7%)
-   **Sentiment analysis** scores for each text response
-   **Video transcriptions** alongside written responses for several questions

### Key Question Categories:

-   **AI Experience**: Personal relationships with AI technology
-   **Sectoral Impact**: Where AI helps or harms different industries
-   **Creative Impact**: AI's role in arts, culture, and creative expression
-   **Community Benefits**: How AI can help BC communities thrive
-   **Concerns & Worries**: Fears about AI's societal impact
-   **Governance**: Advice for BC leaders and policymakers
-   **Indigenous Perspectives**: Role of Indigenous communities in AI development
-   **Future Vision**: What matters most for BC's AI future

## Repository Structure

### Main Visualizations

-   [`index.html`](index.html) - Interactive dashboard with all 16 visualizations
-   [`tsne_visualizations_3d_silhouette_focus_sized/`](tsne_visualizations_3d_silhouette_focus_sized/) - Final 3D visualizations (primary output)

### Processing Scripts

1. [`embeddings_of_data.py`](embeddings_of_data.py) - Generates embeddings for all text columns using OpenAI API
2. [`visualize_embeddings.py`](visualize_embeddings.py) - Basic 2D t-SNE visualization
3. [`visualize_embeddings_with_clusters.py`](visualize_embeddings_with_clusters.py) - Adds K-means clustering
4. [`visualizations_3d_silhouette_focus_sized.py`](visualizations_3d_silhouette_focus_sized.py) - **Final script** - generates the complete 3D visualizations with optimal clustering and frequency-based sizing

### Configuration & Documentation

-   [`requirements.txt`](requirements.txt) - Python dependencies
-   [`.env.template`](.env.template) - Template for OpenAI API key
-   [`README.md`](README.md) - This documentation

### Generated Data

-   `bc_ai_data_with_embeddings.csv` - Original survey data augmented with embeddings (generated during processing, ~600MB)

## Getting Started

### 1. Environment Setup

```bash
# Clone this repository
git clone [repository-url]
cd bc_ai_hackathon_round_3

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure OpenAI API

This project requires OpenAI API access for embedding generation.

1. Copy the environment template:

    ```bash
    copy .env.template .env
    ```

2. Edit `.env` and add your OpenAI API key:
    ```
    OPENAI_API_KEY="sk-your-actual-api-key-here"
    ```

> **Cost Note**: Generating embeddings for all 16 questions costs approximately $1-2 USD.

### 3. Generate Embeddings

Run the embedding script to process all text responses:

```bash
python embeddings_of_data.py
```

This will:

-   Load the BC AI survey data
-   Generate embeddings for all 16 text columns
-   Save results to `bc_ai_data_with_embeddings.csv`
-   Take 5-10 minutes depending on API response times

### 4. Create Final Visualizations

Generate the complete 3D interactive visualizations:

```bash
python visualizations_3d_silhouette_focus_sized.py
```

This will:

-   Perform t-SNE dimensionality reduction to 3D
-   Apply optimal clustering with silhouette score optimization
-   Aggregate identical responses with frequency-based sizing
-   Generate 16 interactive HTML files
-   Save everything to `tsne_visualizations_3d_silhouette_focus_sized/`

### 5. Explore the Results

Open [`index.html`](index.html) in your web browser to access the interactive dashboard with all visualizations.

## Visualizations Overview

Each visualization represents one open-ended question from the BC AI Survey:

| Visualization                 | Question Focus            | Key Insights                                        |
| ----------------------------- | ------------------------- | --------------------------------------------------- |
| 🎨 **AI & Art Perspectives**  | Creative impact of AI     | Artist concerns vs. enhancement views               |
| 🏥 **AI Helping Communities** | Community benefits        | Healthcare, education, accessibility themes         |
| ⚡ **AI Superpowers**         | Ideal AI capabilities     | Practical problem-solving vs. transformative change |
| 😰 **AI Impact Worries**      | Societal concerns         | Job displacement, privacy, control issues           |
| 🏛️ **Advice to BC Leaders**   | Governance guidance       | Regulation, ethics, innovation balance              |
| 🔮 **BC's AI Future**         | Provincial priorities     | Economic growth vs. social responsibility           |
| 🏔️ **Indigenous Involvement** | Indigenous perspectives   | Sovereignty, representation, traditional knowledge  |
| ✅ **Positive AI Sectors**    | Beneficial applications   | Healthcare, education, environmental solutions      |
| ❌ **Negative AI Sectors**    | Harmful applications      | Social media, weapons, surveillance concerns        |
| 🛡️ **Most Trustworthy AI**    | Trusted applications      | Medical AI, research tools, accessibility aids      |
| 🦎 **AI Experience Stories**  | Personal AI relationships | Learning curves, skepticism, adaptation             |

## Technical Details

### Embedding Model

-   **Model**: OpenAI `text-embedding-3-large`
-   **Dimensions**: 3,072
-   **Advantages**: High semantic accuracy, robust handling of diverse text

### Clustering Algorithm

-   **Method**: K-Means with Silhouette Score optimization
-   **Range**: 2-30 clusters tested per question
-   **Selection**: Highest silhouette score determines optimal k

### Visualization Framework

-   **Library**: Plotly for interactive 3D scatter plots
-   **Features**: Rotation, zoom, pan, hover details
-   **Sizing**: Cube root of frequency (volume proportional to count)
-   **Colors**: Cluster-based with automatic palette

### Performance Considerations

-   **Processing Time**: ~10 minutes for embeddings, ~5 minutes for visualizations
-   **Memory Usage**: ~600MB for embedding storage
-   **Browser Compatibility**: Modern browsers with WebGL support

## Future Enhancements

1. **VR/AR Integration**: Immersive exploration in virtual reality environments
2. **Real-time Filtering**: Dynamic filtering by demographics, sentiment, or keywords
3. **Comparative Analysis**: Side-by-side visualization of different demographic groups
4. **AI-Generated Insights**: LLM-powered summaries and pattern identification
5. **Temporal Analysis**: Track opinion evolution if longitudinal data becomes available

## Citation & Usage

This project is open source and available for research, education, and policy analysis. When using this work, please cite:

```
BC AI Survey Data Storytelling: 3D Semantic Visualizations
[Your Name/Organization], 2025
Interactive semantic analysis of 1,001 British Columbian perspectives on artificial intelligence
```

## Contact & Collaboration

For questions, collaboration opportunities, or access to additional analysis:

-   **Issues**: Use GitHub Issues for technical questions
-   **Data Requests**: Contact [data-team@your-org.com]
-   **Media Inquiries**: Contact [media@your-org.com]

---

**🎯 Mission**: Making complex public opinion data accessible through innovative visualization, ensuring every British Columbian voice in this survey can be discovered and understood.
