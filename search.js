class SearchEngine {
    constructor() {
        this.wikiApiUrl = "https://pt.wikipedia.org/w/api.php";
    }

    async search(query) {
        // 1. Busca na Wikipédia (Prioritária)
        const wikiResults = await this.searchWikipedia(query);
        
        // 2. Mock de Busca Web Geral (DuckDuckGo API requer proxy/backend para CORS)
        // Simularemos resultados web complementares
        const webResults = this.mockWebSearch(query);

        return [...wikiResults, ...webResults].slice(0, 15); // Limite 7-20
    }

    async searchWikipedia(query) {
        const params = new URLSearchParams({
            action: 'query',
            list: 'search',
            srsearch: query,
            format: 'json',
            origin: '*' // Crucial para CORS
        });

        try {
            const response = await fetch(`${this.wikiApiUrl}?${params}`);
            const data = await response.json();
            
            if (!data.query || !data.query.search) return [];

            return data.query.search.map(item => ({
                source: 'Wikipedia',
                title: item.title,
                snippet: item.snippet.replace(/<[^>]*>?/gm, ''), // Remove HTML
                url: `https://pt.wikipedia.org/wiki/${encodeURIComponent(item.title)}`
            }));
        } catch (e) {
            console.error("Erro na Wiki:", e);
            return [];
        }
    }

    mockWebSearch(query) {
        // Simula busca em sites de notícias/técnicos
        return [
            {
                source: 'Web',
                title: `Últimas notícias sobre ${query}`,
                snippet: `Resultados recentes da web indicam discussões ativas sobre ${query} em fóruns e sites especializados.`,
                url: '#'
            },
            {
                source: 'Wiktionary',
                title: `Definição de ${query}`,
                snippet: `Definição formal e etimologia encontrada no Wikcionário.`,
                url: `https://pt.wiktionary.org/wiki/${query}`
            }
        ];
    }
}

const searchEngine = new SearchEngine();