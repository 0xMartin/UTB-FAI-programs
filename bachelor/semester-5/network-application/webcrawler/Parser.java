import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLConnection;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

public class Parser implements Runnable {

	static String CSS_QUERY_SELECTOR = "a[href], frame[src], iframe[src]";

	private final WebCrawler crawler;
	private final WebCrawler.URLinfo urlInfo;
	private String pageCharSet;

	public Parser(WebCrawler crawler, WebCrawler.URLinfo urlInfo) {
		this.crawler = crawler;
		this.urlInfo = urlInfo;
		this.crawler.getVisitedURIs().add(urlInfo.uri.toString());
	}

	@Override
	public void run() {
		if (urlInfo.uri.toString().isEmpty())
			return;

		try {
			System.out.printf("Start downloading page [%s][depth: %d] %s\n", this.toString(), this.urlInfo.depth,
					this.urlInfo.uri);

			// kodovani stranky
			this.pageCharSet = Parser.getCharSet(urlInfo.uri);
			// stazeni stranky
			/*
			 * Document doc = Jsoup.parse(new URL(urlInfo.uri.toString()).openStream(),
			 * "UTF-8", urlInfo.uri.toString());
			 */
			Document doc = Jsoup.connect(validURL(urlInfo.uri.toString())).get();

			System.out.printf("Analysating [%s][depth: %d] %s\n", this.toString(), this.urlInfo.depth,
					this.urlInfo.uri);

			// analyza + hledani linku
			try {
				this.crawler.getAnalyzer().analyze(doc, this.pageCharSet);
				if (urlInfo.depth + 1 <= this.crawler.getMaxDetph()) {
					findLinks(doc, urlInfo.depth + 1);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}

		} catch (IOException e1) {
			// e1.printStackTrace();
			System.out.printf("Page download failed [%s][depth: %d] %s\n", this.toString(), this.urlInfo.depth,
					this.urlInfo.uri);
		}
	}

	/**
	 * Na webove strance nalezne dalsi odkazi
	 * 
	 * @param doc   JSOUP Document
	 * @param depth Aktualni hloubka analyzi
	 */
	public void findLinks(Document doc, int depth) {
		System.out.printf("Finding links [%s][depth: %d] %s\n", this.toString(), this.urlInfo.depth, this.urlInfo.uri);

		Elements newsHeadlines = doc.select(CSS_QUERY_SELECTOR);
		for (Element headline : newsHeadlines) {
			// System.out.printf("%s\t%s\n", headline.tagName(), headline.absUrl("href"));
			try {
				this.crawler.getURLQueue().add(new WebCrawler.URLinfo(headline.absUrl("href"), depth));
			} catch (URISyntaxException e) {
				e.printStackTrace();
			}
		}
	}

	private static String getCharSet(URI uri) {
		String charset = "UTF-8";
		try {
			URLConnection conn = uri.toURL().openConnection();
			String type = conn.getContentType();
			int encodingIndex = type.indexOf(';');
			if (encodingIndex >= 0) {
				String encoding = type.substring(type.indexOf(';') + 2);
				charset = encoding.substring(encoding.indexOf('=') + 1);
			}
		} catch (Exception e) {
		}
		return charset;
	}

	private static String validURL(String urlStr) {
		try {
			URL url = new URL(URLDecoder.decode(urlStr, StandardCharsets.UTF_8.toString()));
			URI uri = new URI(url.getProtocol(), url.getUserInfo(), url.getHost(), url.getPort(), url.getPath(),
					url.getQuery(), url.getRef());
			return uri.toString();
		} catch (URISyntaxException | UnsupportedEncodingException | MalformedURLException e) {
			return null;
		}
	}

	/**************************************************************************************************/
	// LOCAL CLASSES
	/**************************************************************************************************/
	
	public static interface Analyzer {
		public void analyze(Document doc, String charSet) throws Exception;
		public void printResult(Object[] args);
	}
	
}
