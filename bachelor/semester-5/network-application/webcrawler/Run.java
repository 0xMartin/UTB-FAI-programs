import java.net.URISyntaxException;

public class Run {
	
	public static void main(String[] args) {
		String url = "";
		int maxDepth = 0;
		int threadCount = 100;
		int wordCount = 20;

		if (args.length > 0) {
			if (args[0].startsWith("--help")) {
				System.out.printf("Pouziti: WebCrawler [URL] [MAX_DEPTH] [THREAD_COUNT] [WORD_COUNT]\n"
						+ "URL - Adresa analyzovane webove stranky \n"
						+ "MAX_DEPTH - Maximalni hloubka analyzi (default: 0)\n"
						+ "THREAD_COUNT - Maximalni pocet vlaken (default: 100)\n"
						+ "WORD_COUNT - Pocet nejcetnejsich slov (default: 20)");
				return;
			}
			// URL
			url = args[0];
			// MAX_DEPTH
			if (args.length > 1) {
				try {
					maxDepth = Integer.parseInt(args[1]);
				} catch (NumberFormatException ex) {
					System.err.println("Hodnota vstupniho parametru [MAX_DEPTH] je neplatna");
					return;
				}
			}
			// THREAD_COUNT
			if (args.length > 2) {
				try {
					threadCount = Integer.parseInt(args[2]);
				} catch (NumberFormatException ex) {
					System.err.println("Hodnota vstupniho parametru [THREAD_COUNT] je neplatna");
					return;
				}
			}
			// WORD_COUNT
			if (args.length > 3) {
				try {
					wordCount = Integer.parseInt(args[3]);
				} catch (NumberFormatException ex) {
					System.err.println("Hodnota vstupniho parametru [WORD_COUNT] je neplatna");
					return;
				}
			}
		} else {
			System.err.println("URL nebyl specifikovan");
			return;
		}

		// run WebCrawler

		try {
			System.out.printf("WebCrawler\n"
					+ "======================================================================================\n"
					+ "URL: %s\n" + "MAX_DEPTH: %d\n" + "THREADS: %d\n"
					+ "======================================================================================\n", url,
					maxDepth, threadCount);
			Thread.sleep(1000);

			WebCrawler webCrawler = new WebCrawler(threadCount, new WordFrequencyAnalyzer());
			webCrawler.parse(url, maxDepth);
			webCrawler.printResult(new Object[] {wordCount});
			System.exit(0);
		} catch (URISyntaxException | InterruptedException e) {
			e.printStackTrace();
		}
	}

}
