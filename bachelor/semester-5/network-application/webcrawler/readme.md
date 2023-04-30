# WebCrawler

[GO BACK](https://github.com/0xMartin/UTB-FAI-programs)

This repository contains a Java implementation of a web crawler, which is a command-line application that allows the user to set a starting URL, a number of threads, and a maximum crawling depth. Parallel processing is used to maximize performance, and a ThreadPoolExecutor is utilized. The application makes use of org.jsoup to analyze web pages.

## Usage

To run the web crawler, use the following command:

```
java -jar WebCrawler.jar [URL] [MAX_DEPTH] [THREAD_COUNT] [WORD_COUNT]
```
The available options are:

* URL: the starting URL for crawling
* MAX_DEPTH, the maximum depth to crawl
* THREAD_COUNT: the number of threads to use
* WORD_COUNT: number of printed worlds that are most commen on all pages 
    * WordFrequencyAnalyzer.java - can be replaced by differt module