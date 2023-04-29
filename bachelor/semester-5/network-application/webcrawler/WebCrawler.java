import java.net.URI;
import java.net.URISyntaxException;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

public class WebCrawler {

	private final Parser.Analyzer analyzer;
	private final ThreadPoolExecutor executor;
	private final List<URLinfo> urlQueue;
	private final Set<String> visitedURIs;
	private int max_depth;

	public WebCrawler(int threadCount, Parser.Analyzer analyzer) {
		this.executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(Math.max(1, threadCount));
		this.urlQueue = Collections.synchronizedList(new LinkedList<URLinfo>());
		this.visitedURIs = Collections.newSetFromMap(new ConcurrentHashMap<String, Boolean>());
		this.analyzer = analyzer;
	}

	public List<URLinfo> getURLQueue() {
		return this.urlQueue;
	}

	public Set<String> getVisitedURIs() {
		return this.visitedURIs;
	}

	public int getMaxDetph() {
		return this.max_depth;
	}
	
	public Parser.Analyzer getAnalyzer() {
		return this.analyzer;
	}

	/**
	 * Analyzuje webovou stranku
	 * @param url - URL adresa webove stranky
	 * @param max_depth - maximalni hloubka analyzi
	 * @throws URISyntaxException
	 */
	public void parse(String url, int max_depth) throws URISyntaxException {
		long start = System.currentTimeMillis();

		urlQueue.add(new URLinfo(url, 0));
		this.max_depth = max_depth;

		URLinfo urlInfo;
		while (!urlQueue.isEmpty() || this.executor.getActiveCount() > 0) {
			if (!urlQueue.isEmpty()) {
				urlInfo = urlQueue.get(0);
				//zavola parser pro danou URL pokud jeste nebyla navstivena
				if (!this.visitedURIs.contains(urlInfo.uri.toString()) && !urlInfo.uri.isOpaque()) {
					this.executor.execute(new Parser(this, urlInfo));
				}
				urlQueue.remove(0);
			} else {
				try {
					Thread.sleep(1);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}

		System.out.println(String.format("==[FINISHED]===[ELAPSED TIME: %.3f s]====================================================",
				(System.currentTimeMillis() - start) / 1e3).substring(0, 85));
	}

	/**
	 * Vypise vysledek analyzy
	 * @param args - Argumenty predane tride analyzatoru
	 */
	public void printResult(Object[] args) {
		this.analyzer.printResult(args);
	}

	/**************************************************************************************************/
	// LOCAL CLASSES
	/**************************************************************************************************/

	public static class URLinfo {
		URI uri;
		int depth;

		public URLinfo(String url, int depth) throws URISyntaxException {
			this.uri = new URI(url);
			this.depth = depth;
		}
	}

}
