<p><b>Bigram analysis of texts</b><br>
<br>
This project is part of an NLP course at Universität Würzburg.<br>
The task is to analyze a text for the occurence of typical bigrams, i.e. word-pairs. The procedure is shown using the example of the entire text of the German novel "Effie Briest" by Theodor Fontane. For example, the word pair "Tante Therese" may be expected to occur very often, as this is the name of a main personage; or "weites Feld", because this is an expression the persons tend to use repeatedly. Analyses of this kind may allow first statements on context, relationships and sentiment of a given text.<br>
Statistical analysis will follow the work of Dunning in <a href="https://www.google.de/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwilmsD7weHXAhVLC-wKHWVoB9IQFggwMAE&url=http%3A%2F%2Faclweb.org%2Fanthology%2FJ93-1003&usg=AOvVaw255tnabSLrnqmqx4QJcoKu">"Accurate Methods for the Statistics of Surprise and Coincidence"</a>.<br>
  
Run the script on a text file that you would like to know the most significant bigrams for.<br>
The text should be encoded in UTF-8:<br>
    $ python main.py path-to-text-file<br>
Try this on a sample text in the resources directory, the German novel "Effie Briest" by Theodor Fontane:<br>
    $ python main.py effie.txt<br>

