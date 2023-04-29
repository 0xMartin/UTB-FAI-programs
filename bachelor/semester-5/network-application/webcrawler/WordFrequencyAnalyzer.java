import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jsoup.nodes.Document;


public class WordFrequencyAnalyzer implements Parser.Analyzer {

	private final Map<String, Word> wordCounter;
	
	public WordFrequencyAnalyzer() {
		this.wordCounter = Collections.synchronizedMap(new HashMap<String, Word>());
	}
	
	@Override
	public void analyze(Document doc, String charSet) throws Exception {
		String text = doc.body().text();
		BufferedReader reader = new BufferedReader(
				new InputStreamReader(new ByteArrayInputStream(text.getBytes()), charSet));
		String line;
		while ((line = reader.readLine()) != null) {
			String[] words = line.split("\\s+");
			for (String word : words) {
				Word w = this.wordCounter.get(word);
				if (w == null) {
					w = new Word(word);
					this.wordCounter.put(word, w);
				}
				w.count++;
			}
		}
		reader.close();
	}
	
	@Override
	public void printResult(Object[] args) {
		int words = Integer.parseInt(args[0].toString());
		System.out.println("==[RESULT]===========================================================================");		
		List<Word> sortedWords = new ArrayList<Word>(this.wordCounter.values());
		Collections.sort(sortedWords);
		
		System.out.println("ID\tCount\tWord");
		int i = 0;
		for (Word word : sortedWords) { 
			i++;
			if (i > words)
				break;
			System.out.printf("[%d]\t%d\t%s\n", i, word.count, word.word);
		}
		System.out.println("======================================================================================");
	}
		
}

class Word implements Comparable<Word> {
	String word;
	int count;

	public Word(String word) {
		this.word = word;
	}

	@Override
	public int compareTo(Word b) {
		return b.count - count;
	}
}
