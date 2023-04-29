import time
import requests
from bs4 import BeautifulSoup
import numpy as np


class PageRank:
    # list with visited URLs, element structure: {url_from, url_to}
    urls = list()

    startURL: str = ""
    searchDepth: int = 0

    def __init__(self, startURL: str, searchDepth: int) -> None:
        self.startURL = startURL
        self.searchDepth = searchDepth

    def runWebCrawler(self):
        self.urls.clear()  # clear list with visited URLs

        URLqueue = list()  # element structure: {url, depth}
        URLqueue.append([self.startURL, 0])  # start URL with index 0

        for depth in range(self.searchDepth):
            for url in URLqueue:
                # URL from queue must be of same depth as current depth
                if url[1] != depth:
                    continue

                # find links on page
                print("Page:", url[0])
                links = self.__findLinks(url[0])

                # remove duplicite links from page
                links = list(set(links))
                print("Links count: ", len(links))

                # add links to queue and visited urls
                ndepth = depth + 1
                for l in links:
                    URLqueue.append([l, ndepth])
                    self.urls.append([url[0], l])

    def __findLinks(self, url: str) -> list():
        out = list()

        # find links
        reqs = requests.get(url)
        soup = BeautifulSoup(reqs.text, 'html.parser')
        for link in soup.find_all('a'):
            l = link.get('href')
            if l is not None:
                if l.startswith("https://") or l.startswith("http://"):
                    # absolute URL
                    out.append(l)
                elif not l.startswith("#"):
                    # relative URL
                    if url.endswith("/") and l.startswith("/"):
                        r = url[:-1] + l
                    else:
                        r = url + l
                    out.append(r)

        return out

    def computeResult(self, iters: int, beta: float = 0.85) -> list():
        print("Page Rank start ...")

        # all pages
        pages = list()
        for url in self.urls:
            pages.append(str(url[0]))
            pages.append(str(url[1]))
        pages = list(set(pages))

        # page count
        N = len(pages)
        print("N =", N)

        # identity matrix
        E = np.identity(N)

        # buid M matrix
        M = np.zeros((N, N))
        for col_index in range(N):
            col_url = pages[col_index]
            total = self.__countURLs_to(col_url)
            if total == 0:
                continue
            for row_index in range(N):
                p = self.__countURLs(pages[row_index], col_url) / total
                M[row_index][col_index] = p

        # rank matrix
        rank = np.ones(N) / N
        # page rank alghoritm
        A = beta * M + (1.0 - beta) * (1.0/N) 
        for _ in range(iters):
            rank = A @ rank

        # get rank from matrix for each page
        page_ranks = list()
        for i in range(N):
            page_ranks.append([pages[i], rank[i]])

        # sort
        page_ranks.sort(key=lambda e: e[1], reverse=True)

        print("Page Rank finished ...")
        return page_ranks

    def __countURLs_to(self, toURL):
        """
        The number of links that point to the defined URL "page with URL: toURL"
        """
        cnt = 0
        for u in self.urls:
            if u[1] == toURL:
                cnt += 1
        return cnt

    def __countURLs(self, fromURL, toURL):
        """
        The number of links that point from defined page to defined page
        """
        cnt = 0
        for u in self.urls:
            if u[0] == fromURL and u[1] == toURL:
                cnt += 1
        return cnt


if __name__ == '__main__':
    # test
    start = time.time()

    url = input("Start URL (https://openai.com/):")
    if len(url) == 0:
        url = "https://openai.com/"
    depth = input("Search depth (2):")
    if len(depth) == 0:
        depth = "2"

    pr = PageRank(url, int(depth))
    pr.runWebCrawler()
    res = pr.computeResult(50)

    end = time.time()
    print("Total time:", end - start, "s")

    # print output
    print("\nPage ranks:")
    sum_check = 0.0
    for i, r in enumerate(res):
        sum_check += r[1]
        if i < 30:
            print("[%d] %s - %f" % (i + 1, r[0], r[1]))
    print("Sum =", sum_check)

    # save to file
    opt = input("Save to file? [y/N]:")
    if(opt == "y" or opt == "Y"):
        name = input("Name of your file:")
        f = open(name + ".csv", "w")
        for i, r in enumerate(res):
            f.write("%d;%s;%f\n" % (i + 1, r[0], r[1]))
        f.close()
