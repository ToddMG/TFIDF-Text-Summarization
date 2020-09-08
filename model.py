
import pandas as pd
import re
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

def _corpus_splitter(raw_text):
    split_corpus = re.split('[?!.:]', raw_text)

    return split_corpus

def _vectorizer(split_corpus):
    vectorizer = TfidfVectorizer(use_idf=True, stop_words='english')
    tf_idf = vectorizer.fit_transform(split_corpus)
    df_idf = pd.DataFrame(vectorizer.idf_, index=vectorizer.get_feature_names(), columns=["idf_weights"])
    s_idf = df_idf['idf_weights']

    term_weights = {}
    vocab = s_idf.index.to_list()
    i = 0
    for word in vocab:
        term_weights[word] = s_idf.iloc[i]
        i += 1

    return term_weights


def _score_document(doc, term_weights):
    """
    :param doc_matrix: Document matrix with terms and their respective scores
    :return: Document's total term score
    """
    doc_score = 0
    for word in doc.split():
        if word in term_weights.keys():
            doc_score += term_weights[word]
            # print(doc_score)

    return doc_score


def _summarizer(corpus, percent):
    split_corpus = _corpus_splitter(corpus)
    term_weights = _vectorizer(split_corpus)
    scored_docs = {}

    for doc in split_corpus:
        doc_score = _score_document(doc, term_weights)

        scored_docs[doc_score] = doc

    percentile = np.percentile(list(scored_docs.keys()), percent)

    return scored_docs, percentile


if __name__ == '__main__':

    corpus = """Unsupervised learning is the ability to find patterns in a stream of input, without requiring a human to label the inputs first. Supervised learning includes both classification and numerical regression, which requires a human to label the input data first. Classification is used to determine what category something belongs in, and occurs after a program sees a number of examples of things from several categories. Regression is the attempt to produce a function that describes the relationship between inputs and outputs and predicts how the outputs should change as the inputs change. Both classifiers and regression learners can be viewed as "function approximators" trying to learn an unknown (possibly implicit) function; for example, a spam classifier can be viewed as learning a function that maps from the text of an email to one of two categories, "spam" or "not spam". Computational learning theory can assess learners by computational complexity, by sample complexity (how much data is required), or by other notions of optimization. In reinforcement learning the agent is rewarded for good responses and punished for bad ones. The agent uses this sequence of rewards and punishments to form a strategy for operating in its problem space."""
    doc = 'Unsupervised learning is the ability to find patterns in a stream of input, without requiring a human to label the inputs first'

    corpus2 = """
    Disney is under fire for shooting its new film Mulan in parts of China where the government is accused of serious human rights abuses.

The final credits thank a government security agency in Xinjiang province, where about 1m people - mostly Muslim Uighurs - are thought to be detained.

The film was already the target of a boycott after its lead actress backed a crackdown on Hong Kong protesters.

Disney has not commented on the row over the locations and the credits.

China says its detention camps in Xinjiang are necessary to improve security.

The live-action film, which is one of the biggest releases of the year, is a remake of the 1998 animated story of a young girl who takes her father's place in the army.

But fans in some Asian countries called for a boycott after Chinese-born actress Liu Yifei made comments supporting Hong Kong's police who have been accused of violence against pro-democracy protesters in recent months.

Then, on Monday, social media users noticed that in the credits Disney thanked a number of government entities in Xinjiang, including the public security bureau in the city of Turpan and the "publicity department of CPC Xinjiang Uyghur Autonomy Region Committee".

The public security bureau in Turpan is tasked with running China's "re-education" camps where Uighurs are held in detention, China expert Adrian Zenz told the BBC.

The "publicity department" named by Disney is responsible for producing state propaganda in the region, he adds.

It is believed that one million Uighur people have been forcibly detained in the high-security prison camps in recent years.

Leaked documents and testimonies from camp survivors reveal that inmates are locked up, indoctrinated and punished, claims which China has dismissed as "fake news".

In 2018 a BBC investigation found evidence of security compounds built in the desert in Xinjiang.

A model's video gives rare glimpse inside internment
Who are the Uighurs?
Detained for beards, veils and internet browsing
Mr Zenz described Disney as "an international corporation profiteering in the shadow of concentration camps".

The World Uygar Congress tweeted "in the new Mulan, Disney thanks the public security bureau in Turpan, which has been involved in the internment camps in East Turkistan."

Activist Shawn Zhang also criticised the company, writing "how many thousands of Uighur were put into camps by Turpan Bureau of Public Security when filming Mulan there?"

Turpan was the site of the first "re-education camps" where Uighur women wearing veils or men wearing beards were detained, Mr Zenz explained. The public security bureau is also responsible for managing construction of the camps and hiring police to staff them, he added.

The earliest evidence of "re-education" work of Uighurs in Turpan is from August 2013, Mr Zenz claims.

In June, he issued a report which uncovered evidence that China was forcing Uighur women to be sterilised or fitted with contraceptive devices, a practice that China denies.

China says it is fighting the "three evil forces" of separatism, terrorism, and extremism in Xinjiang and says the camps are voluntary schools for anti-extremism training.

China's hidden camps
In 2017 Mulan director Niki Caro posted photos on Instagram from the capital of Xinjiang. The production team behind the film also told the Architectural Digest magazine that they spent months in Xinjiang to research filming locations.

Hong Kong pro-democracy activist Joshua Wong has also condemned Disney, tweeting that viewers watching Mulan are "potentially complicit in the mass incarceration of Muslim Uighurs"."""

    corpus3 = """
    Trump Emerges as Inspiration for Germany’s Far Right
Among German conspiracy theorists, ultranationalists and neo-Nazis, the American president is surfacing as a rallying cry, or even as a potential “liberator.”

[Image: Coronavirus skeptics and far-right supporters during a protest in Berlin on Aug. 29.]

Credit...Omer Messinger/Getty Images
By Katrin Bennhold

Sept. 7, 2020
BERLIN — Just before hundreds of far-right activists recently tried to storm the German Parliament, one of their leaders revved up the crowd by conjuring President Trump.

“Trump is in Berlin!” the woman shouted from a small stage, as if to dedicate the imminent charge to him.

She was so convincing that several groups of far-right activists later showed up at the American Embassy and demanded an audience with Mr. Trump. “We know he’s in there!” they insisted.

Mr. Trump was neither in the embassy nor in Germany that day — and yet there he was. His face was emblazoned on banners, T-shirts and even on Germany’s pre-1918 imperial flag, popular with neo-Nazis in the crowd of 50,000 who had come to protest Germany’s pandemic restrictions. His name was invoked by many with messianic zeal.

It was only the latest evidence that Mr. Trump is emerging as a kind of cult figure in Germany’s increasingly varied far-right scene.

“Trump has become a savior figure, a sort of great redeemer for the German far right,” said Miro Dittrich, an expert on far-right extremism at the Berlin-based Amadeu-Antonio-Foundation.

Germany — a nation generally supportive of a government that has handled the pandemic better than most — may seem an unlikely place for Mr. Trump to gain such a status. Few Western nations have had a more contentious relationship with Mr. Trump than Germany, whose leader, Chancellor Angela Merkel, a pastor’s daughter and scientist, is his opposite in terms of values and temperament. Opinion polls show that Mr. Trump is deeply unpopular among a broad majority of Germans.

But his message of disruption — his unvarnished nationalism and tolerance of white supremacists coupled with his skepticism of the pandemic’s dangers — is spilling well beyond American shores, extremism watchers say.

In a fast-expanding universe of disinformation, that message holds real risks for Western democracies, they say, blurring the lines between real and fake news, allowing far-right groups to extend their reach beyond traditional constituencies and seeding the potential for violent radicalization.

Mr. Trump’s appeal to the political fringe has now added a new and unpredictable element to German politics at a time when the domestic intelligence agency has identified far-right extremism and far-right terrorism as the biggest risks to German democracy.

The authorities have only recently woken up to a problem of far-right infiltration in the police and military. Over the past 15 months, far-right terrorists killed a regional politician on his front porch near the central city of Kassel, attacked a synagogue in the eastern city of Halle and shot dead nine people of immigrant descent in the western city of Hanau. Mr. Trump featured in the manifesto of the Hanau killer, who praised his “America First” policy.

In Germany, as in the United States, Mr. Trump has become an inspiration to these fringe groups. Among them are not only long-established hard-right and neo-Nazi movements, but also now followers of QAnon, the internet conspiracy theory popular among some of Mr. Trump’s supporters in the United States that hails him as a hero and liberator.

Germany’s QAnon community, barely existent when the pandemic first hit in March, may now be the biggest outside the United States along with Britain, analysts who track its most popular online channels say.

Image
[Image: Riot police officers detaining a woman wearing a QAnon T-shirt in Berlin last weekend.]

Credit...Sean Gallup/Getty Images
Matthias Quent, an expert on Germany’s far right and the director of an institute that studies democracy and civil society, calls it the “Trumpification of the German far right.”

“Trump has managed to attract different milieus, and that’s what we’re seeing here, too,” Mr. Quent said. “We have everything from anti-vaxxers to neo-Nazis marching against corona measures. The common denominator is that it’s people who are quitting the mainstream, who are raging against the establishment.”

Trump, he added, “is the guy fighting the liberal-democratic establishment.”

For some on the far-right fringes, Mr. Trump’s message has been especially welcome at a time when Germany’s homegrown nativist party, the Alternative for Germany, or AfD, is struggling to exploit the pandemic and has seen its support dip to around 10 percent, experts say.

Nationalist populists in Germany have long welcomed the presence of one of their own, as they see it, in the White House. Mr. Trump’s language and ideology have helped legitimize theirs.

The AfD has repeatedly paraphrased Mr. Trump by calling for a “Germany first” approach. But the president is popular in more extremist circles, too. Caroline Sommerfeld, a prominent ideologue of a contingent known as the “new right” with close links to the extremist Generation Identity movement, said she had popped open a bottle of champagne when Mr. Trump won the 2016 election.

The QAnon phenomenon has added a new kind of fuel to that fire.

QAnon followers argue that Mr. Trump is fighting a “deep state” that not only controls finance and power, but also abuses and murders children in underground prisons to extract a substance that keeps its members young. German followers contend that the “deep state” is global, and that Ms. Merkel is part of it. Mr. Trump, they say, will liberate Germany from the Merkel “dictatorship.”

Image
Credit...Al Drago for The New York Times
The far-right magazine Compact, which has printed Mr. Trump’s speeches for its readers, had a giant Q on its latest cover and held a “Q-week” on its video channel, interviewing far-right extremists like Björn Höcke. On the streets of Berlin last weekend there were Q flags and T-shirts and several banners inscribed with “WWG1WGA,” a coded acronym for Q’s hallmark motto, “Where we go one, we go all.”

Hard numbers are difficult to discern, with followers often subscribing to accounts on different platforms, analysts say. NewsGuard, a U.S.-based disinformation watchdog, found that across Europe, accounts on YouTube, Facebook and Telegram promoting the QAnon conspiracy counted 448,000 followers.

In Germany alone, the number of followers of QAnon-related accounts has risen to more than 200,000, Mr. Dittrich said. The largest German-language QAnon channel on YouTube, Qlobal-Change, has over 17 million views and has quadrupled its following on Telegram to over 124,000 since the coronavirus lockdown in March, he said.

“There is a huge Q community in Germany,” Mr. Dittrich said, with new posts and memes that dominate the message boards in the United States immediately translated and interpreted into German.

The fusion of the traditional far right with the QAnon crowd was something new, Mr. Quent said. “It’s a new and diffuse kind of populist rebellion that feeds on conspiracy theories and is being supplied with ideology from different corners of the far-right ecosystem,” he said.

One reason the QAnon conspiracy has taken off in Germany, Mr. Dittrich said, is that it is a good fit with local conspiracy theories and fantasies popular on the far right.

One of them is the “great replacement,” which claims that Ms. Merkel and other governments have been deliberately bringing in immigrants to subvert Germany’s ethnic and cultural identity. Another is a purported national crisis called “Day X,” when Germany’s current order will supposedly collapse and neo-Nazis take over.

A third theory is the belief that Germany is not a sovereign country but an incorporated company and occupied territory controlled by globalists.

This belief is held among a faction known as “Reichsbürger,” or citizens of the Reich, who orchestrated the brief storming on Parliament on Aug. 29. They do not recognize Germany’s post-World War II Federal Republic and are counting on Mr. Trump and President Vladimir V. Putin of Russia to sign a “peace treaty” to liberate Germans from their own government.

Another reason for QAnon’s spread is that several German celebrities have become multipliers of the conspiracy, among them a former news presenter and a rapper and former judge on Germany’s equivalent of “American Idol.”

One of the biggest figures spreading the QAnon conspiracy is Attila Hildmann, a vegan-celebrity-chef-turned-far-right-influencer with more than 80,000 followers on the Telegram messaging app. He has appeared at all major coronavirus marches in Berlin, venting against face masks, Bill Gates and the Rothschild family — and appealing to Mr. Trump to liberate Germany.

Image
Credit...Tobias Schwarz/Agence France-Presse — Getty Images
“Trump is someone who has been fighting the global ‘deep state’ for years,” Mr. Hildmann said in an interview this past week. “Trump has become a figure of light in this movement, especially for QAnon, precisely because he fights against these global forces.”

“That’s why the hope for the German national movement, or the liberation movement, lies basically with Q and Trump, because Trump is a figure of light because he shows that you can fight these global powers and that he is victorious,” Mr. Hildmann said.

“The Germans hope that Trump will liberate Germany from the Merkel corona regime,” he said, so that “the German Reich is reactivated.”

Mr. Hildmann’s influence became plain in June, when he mobilized thousands of people to send messages to the U.S. and Russian embassies in Berlin to appeal for help. In the space of a few days, 24,000 tweets had been received by the embassies calling on Mr. Trump and Mr. Putin to “liberate” Germany from Ms. Merkel’s “criminal regime” and prevent “forced vaccination” and “genocide.”

Germany’s domestic intelligence agency has warned of the risk of far-right extremists using the pandemic for their own purposes. This past week the agency’s chief, Thomas Haldenwang, said that “aggressive and disruptive far-right elements” were the driving force behind the protests over coronavirus restrictions.

But extremism experts and lawmakers worry that the security services are not paying close enough attention to the violent potential in the mix of QAnon disinformation campaigns and homegrown far-right ideology.

In the United States, some QAnon believers have been charged with violent crimes, including one accused of murdering a mafia boss in New York last year and another arrested in April after reportedly threatening to kill Joseph R. Biden Jr., who has since become the Democratic presidential candidate. The F.B.I. has warned that QAnon poses a potential domestic terrorism threat.

In Germany, language reminiscent of QAnon was used in the manifesto of the gunman who killed nine people with immigrant roots in the western city of Hanau in February.

“We have already seen that this conspiracy has the potential to radicalize people,” Mr. Dittrich said.

There are an estimated 19,000 Reichsbürger in Germany, about 1,000 of them classified as far-right extremists by the domestic intelligence service. Many of them are armed.

“At a time when some people are determined to destroy democratic discourse with all means possible,” said Konstantin von Notz, a lawmaker and deputy president of the intelligence oversight committee, “we have to take such a phenomenon very seriously.”"""

    summary, percentile = _summarizer(corpus3, 95)

    for item in summary:
        if item >= percentile:
            print(f'{item}: {summary[item]}')