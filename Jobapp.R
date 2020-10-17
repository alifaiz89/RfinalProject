library(shiny)
library(dplyr)
library(textstem)
library(tidyr)
library(tidytext)
library(dplyr)
library(stringr)
library(tm)
library(useful)
library(cluster)
library(MLmetrics)
library(caret)
library(e1071)


# UR4A Final Project App 
library(shiny)

cbind.fill <- function(...){
  nm <- list(...) 
  nm <- lapply(nm, as.matrix)
  n <- max(sapply(nm, nrow)) 
  do.call(cbind, lapply(nm, function (x) 
    rbind(x, matrix(, n-nrow(x), ncol(x))))) 
}

ui <- fluidPage(
  headerPanel('UR4A Final Project App'),
  sidebarPanel(
    textAreaInput(
      'jobdesc',
      "Predict Job Description's Job Group!",
      height = 300
    ),
    actionButton("predictbutton", "Predict!"),
    sliderInput( inputId = 'k',
                 label = 'How many groups do you think there is in the Indeed.com Dataset?',
                 min = 2,
                 max = 20,
                 value = 6)
  ),
  mainPanel(
    span(textOutput('pred_result'), style="color:blue; font-size: 20px"),
    plotOutput('plot1'),
    plotOutput('plot2'),
    plotOutput('plot3'),
    plotOutput('plot4'),
    plotOutput('plot5'),
    plotOutput('plot6'),
    plotOutput('plot7')
    # DT::dataTableOutput('mytable1'),
    # DT::dataTableOutput('mytable2'),
    #DT::dataTableOutput('mytable')
  )
)

server <- function(input, output) {
  load("indeed_tidy.RData")
  # Make target variable first column in dataset, get rid of index and Job_Cat_Index
  d <- d[,c(3, 4:ncol(d))]
  
  # Make target (Job_Category) column name "y"
  names(d)[1] <- "y"
  names(d)
  
  ################################################################################
  # Identify correlated variables in the 'd' dataset that are less than or 
  # greater than 85% and remove them. 
  ################################################################################
  
  # calculate correlation matrix using Pearson's correlation formula
  descrCor <-  cor(d[,2:ncol(d)])                           # correlation matrix
  highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .85) # num Xs with cor > t
  summary(descrCor[upper.tri(descrCor)])                    # summarize the cors
  
  # which columns in your correlation matrix have a correlation greater than some
  # specified absolute cutoff. Find them and remove them
  highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.85)
  filteredDescr <- d[,2:ncol(d)][,-highlyCorDescr] # remove those specific columns
  descrCor2 <- cor(filteredDescr)                  # calculate a new cor matrix
  # summarize those correlations to see if all features are now within our range
  summary(descrCor2[upper.tri(descrCor2)])
  
  # update dataset by removing those filtered vars that were highly correlated
  d <- cbind(d$y, filteredDescr)
  names(d)[1] <- "y"
  
  rm(filteredDescr, descrCor, descrCor2, highCorr, highlyCorDescr)  # clean up
  
  ################################################################################
  # identify linear dependencies and remove them to reduce the issue of 
  # perfect collinearity using the findLinearCombos() function
  ################################################################################
  
  # first save response
  y <- d$y
  
  # create a column of 1s. This will help identify all the right linear combos
  d <- cbind(rep(1, nrow(d)), d[2:ncol(d)])
  names(d)[1] <- "ones"
  
  # identify the columns that are linear combos
  comboInfo <- findLinearCombos(d)
  comboInfo
  
  # remove columns identified that led to linear combos
  #d <- d[, -comboInfo$remove]  #No linear combos was found
  
  # remove the "ones" column in the first column
  d <- d[, c(2:ncol(d))]
  
  # Add the target variable back to our data.frame
  d <- cbind(y, d)
  
  rm(y, comboInfo)  # clean up
  
  ################################################################################
  # remove features with limited variation using nearZeroVar() from the 'd'
  # data.frame 
  ################################################################################
  
  nzv <- nearZeroVar(d, saveMetrics = TRUE)
  head(nzv)
  table(nzv$zeroVar)
  table(nzv$nzv)
  d <- d[, c(TRUE,!nzv$nzv[2:ncol(d)])]
  #d <- d[, c(TRUE,!nzv$zeroVar[2:ncol(d)])]
  
  rm(nzv)
  
  final_keywords <- as.data.frame(names(d[, 2:ncol(d)]))
  names(final_keywords) <- "word"
  write.table(final_keywords, file = "final_keywords.csv", sep = ",", row.names = F)
  
  ################################################################################
  # using preProcess(), standardize your numeric features using a min-max 
  # normalization. 
  ################################################################################
  
  preProcValues <- preProcess(d[,2:ncol(d)], method = c("range"))
  d <- predict(preProcValues, d)
  
  rm(preProcValues)
  
  ################################################################################
  # Create a 80/20 train/test set using createDataPartition(). 
  ################################################################################
  
  set.seed(1234) # set a seed so you can replicate your results
  # identify records that will be used in the training set. Here we are doing a
  # 50/50 train-test split.
  # inTrain <- createDataPartition(y = d$y,   # outcome variable
  #                                p = .50,   # % of training data you want
  #                                list = F)
  
  inTrain <- createDataPartition(y = d$y,   # outcome variable
                                 p = .80,   # % of training data you want
                                 list = F)
  
  # create your partitions
  train <- d[inTrain,]  # training data set
  test <- d[-inTrain,]  # test data set
  
  ################################################################################
  # RUN MULTINOMIAL MODEL ON D DATA
  ################################################################################
  levels(train$y) <- make.names(levels(factor(train$y)))
  levels(train$y)
  
  levels(test$y) <- make.names(levels(factor(test$y)))
  levels(test$y)
  
  # variable and 'X0' is the same as 0 for the Y variable.
  train$y <- relevel(train$y,"Business.Analyst")
  test$y <- relevel(test$y,"Business.Analyst")
  
  
  ctrl <- trainControl(method="cv",     # cross-validation set approach to use
                       number=3,        # k number of times to do k-fold
                       classProbs = T,  # if you want probabilities
                       summaryFunction = multiClassSummary, # for classification
                       allowParallel=T)
  library(MLmetrics)
  
  # train a multinomial model on train set 
  require(nnet)
  multinom.fit <- train(y ~ .,               # model specification
                        data = train,        # train set used to build model
                        method = "multinom",      # type of model you want to build
                        trControl = ctrl,    # how you want to learn
                        metric = "ROC",       # performance measure
                        MaxNWts=84581,
                        maxit = 100
  )
  
  observeEvent(input$predictbutton, {
    final_keywords_exp <- read.csv("final_keywords_exp.csv", header = TRUE)
    final_keywords_exp$word <- as.character(final_keywords_exp$word)
    
    final_keywords_qual <- read.csv("final_keywords_qual.csv", header = TRUE)
    final_keywords_qual$word <- as.character(final_keywords_qual$word)
        
    final_keywords_skill <- read.csv("final_keywords_skill.csv", header = TRUE)
    final_keywords_skill$word <- as.character(final_keywords_skill$word)
    
    final_keybigrams_env <- read.csv("final_keybigrams_env.csv", header = TRUE)
    final_keybigrams_env$bigram <- as.character(final_keybigrams_env$bigram)
    
    final_keybigrams_exp <- read.csv("final_keybigrams_exp.csv", header = TRUE)
    final_keybigrams_exp$bigram <- as.character(final_keybigrams_exp$bigram)
    
    final_keybigrams_qual <- read.csv("final_keybigrams_qual.csv", header = TRUE)
    final_keybigrams_qual$bigram <- as.character(final_keybigrams_qual$bigram)
    
    final_keybigrams_skill <- read.csv("final_keybigrams_skill.csv", header = TRUE)
    final_keybigrams_skill$bigram <- as.character(final_keybigrams_skill$bigram)
    
    library(stopwords)
    data(stop_words)
    
    jobdesc_input <- renderText({ input$jobdesc })
    
    tidy_jobdesc <- data.frame(Description=as.character(jobdesc_input()))
    
    tidy_jobdesc_word_exp <- tidy_jobdesc %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("life cycle"), replacement = "lifecycle")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed(".net"), replacement = "dotnet")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("+"), replacement = "plus")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("C#"), replacement = "csharp")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "’", replacement = "'")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'s", replacement = "")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "n't", replacement = " not")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'ve", replacement = " have")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'re", replacement = " are")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'ll", replacement = " will")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'d", replacement = " would")) %>%
      unnest_tokens(output=word, input=Description) %>%
      anti_join(y = stop_words, by = "word") %>%
      mutate(word = ifelse(word != "data", lemmatize_words(word), word)) %>%
      mutate(word = str_replace(string = word, pattern = "analytics", replacement = "analytic")) %>%
      semi_join(final_keywords_exp) %>%
      count(word) %>%
      pivot_wider(names_from = word, values_from = n,
                  names_prefix = "exp_")
    
    tidy_jobdesc_word_qual <- tidy_jobdesc %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("life cycle"), replacement = "lifecycle")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed(".net"), replacement = "dotnet")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("+"), replacement = "plus")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("C#"), replacement = "csharp")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "’", replacement = "'")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'s", replacement = "")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "n't", replacement = " not")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'ve", replacement = " have")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'re", replacement = " are")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'ll", replacement = " will")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'d", replacement = " would")) %>%
      unnest_tokens(output=word, input=Description) %>%
      anti_join(y = stop_words, by = "word") %>%
      mutate(word = ifelse(word != "data", lemmatize_words(word), word)) %>%
      mutate(word = str_replace(string = word, pattern = "analytics", replacement = "analytic")) %>%
      semi_join(final_keywords_qual) %>%
      count(word) %>%
      pivot_wider(names_from = word, values_from = n,
                  names_prefix = "qual_")
    
    tidy_jobdesc_word_skill <- tidy_jobdesc %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("life cycle"), replacement = "lifecycle")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed(".net"), replacement = "dotnet")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("+"), replacement = "plus")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("C#"), replacement = "csharp")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "’", replacement = "'")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'s", replacement = "")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "n't", replacement = " not")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'ve", replacement = " have")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'re", replacement = " are")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'ll", replacement = " will")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = "'d", replacement = " would")) %>%
      unnest_tokens(output=word, input=Description) %>%
      anti_join(y = stop_words, by = "word") %>%
      mutate(word = ifelse(word != "data", lemmatize_words(word), word)) %>%
      mutate(word = str_replace(string = word, pattern = "analytics", replacement = "analytic")) %>%
      semi_join(final_keywords_skill) %>%
      count(word) %>%
      pivot_wider(names_from = word, values_from = n,
                  names_prefix = "skill_")
    
    tidy_jobdesc_bigram_env <- tidy_jobdesc %>%
      mutate(Description = str_replace_all(string = Description, pattern = "life cycle", replacement = "lifecycle")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed(".net"), replacement = "dotnet")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("+"), replacement = "plus")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("C#"), replacement = "csharp")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("’"), replacement = "'")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'s"), replacement = "")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("n't"), replacement = " not")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'ve"), replacement = " have")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'re"), replacement = " are")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'ll"), replacement = " will")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'d"), replacement = " would")) %>%
      unnest_tokens(output=bigram, input=Description, token = "ngrams", n = 2) %>%
      separate(bigram, c("word1", "word2"), sep = " ", remove = FALSE) %>%
      filter(!word1 %in% stop_words$word & !word2 %in% stop_words$word) %>%
      mutate(word1 = ifelse(word1 != "data", lemmatize_words(word1), word1)) %>%
      mutate(word2 = ifelse(word2 != "data", lemmatize_words(word2), word2)) %>%
      mutate(word1 = str_replace(string = word1, pattern = "analytics", replacement = "analytic")) %>%
      mutate(word2 = str_replace(string = word2, pattern = "analytics", replacement = "analytic")) %>%
      mutate(bigram = paste(word1, word2, sep=" ")) %>%
      semi_join(final_keybigrams_env) %>%
      count(bigram) %>%
      pivot_wider(names_from = bigram, values_from = n,
                  names_prefix = "env_")
    
    tidy_jobdesc_bigram_exp <- tidy_jobdesc %>%
      mutate(Description = str_replace_all(string = Description, pattern = "life cycle", replacement = "lifecycle")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed(".net"), replacement = "dotnet")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("+"), replacement = "plus")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("C#"), replacement = "csharp")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("’"), replacement = "'")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'s"), replacement = "")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("n't"), replacement = " not")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'ve"), replacement = " have")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'re"), replacement = " are")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'ll"), replacement = " will")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'d"), replacement = " would")) %>%
      unnest_tokens(output=bigram, input=Description, token = "ngrams", n = 2) %>%
      separate(bigram, c("word1", "word2"), sep = " ", remove = FALSE) %>%
      filter(!word1 %in% stop_words$word & !word2 %in% stop_words$word) %>%
      mutate(word1 = ifelse(word1 != "data", lemmatize_words(word1), word1)) %>%
      mutate(word2 = ifelse(word2 != "data", lemmatize_words(word2), word2)) %>%
      mutate(word1 = str_replace(string = word1, pattern = "analytics", replacement = "analytic")) %>%
      mutate(word2 = str_replace(string = word2, pattern = "analytics", replacement = "analytic")) %>%
      mutate(bigram = paste(word1, word2, sep=" ")) %>%
      semi_join(final_keybigrams_exp) %>%
      count(bigram) %>%
      pivot_wider(names_from = bigram, values_from = n,
                  names_prefix = "exp_")
    
    tidy_jobdesc_bigram_qual <- tidy_jobdesc %>%
      mutate(Description = str_replace_all(string = Description, pattern = "life cycle", replacement = "lifecycle")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed(".net"), replacement = "dotnet")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("+"), replacement = "plus")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("C#"), replacement = "csharp")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("’"), replacement = "'")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'s"), replacement = "")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("n't"), replacement = " not")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'ve"), replacement = " have")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'re"), replacement = " are")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'ll"), replacement = " will")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'d"), replacement = " would")) %>%
      unnest_tokens(output=bigram, input=Description, token = "ngrams", n = 2) %>%
      separate(bigram, c("word1", "word2"), sep = " ", remove = FALSE) %>%
      filter(!word1 %in% stop_words$word & !word2 %in% stop_words$word) %>%
      mutate(word1 = ifelse(word1 != "data", lemmatize_words(word1), word1)) %>%
      mutate(word2 = ifelse(word2 != "data", lemmatize_words(word2), word2)) %>%
      mutate(word1 = str_replace(string = word1, pattern = "analytics", replacement = "analytic")) %>%
      mutate(word2 = str_replace(string = word2, pattern = "analytics", replacement = "analytic")) %>%
      mutate(bigram = paste(word1, word2, sep=" ")) %>%
      semi_join(final_keybigrams_qual) %>%
      count(bigram) %>%
      pivot_wider(names_from = bigram, values_from = n,
                  names_prefix = "qual_")
    
    tidy_jobdesc_bigram_skill <- tidy_jobdesc %>%
      mutate(Description = str_replace_all(string = Description, pattern = "life cycle", replacement = "lifecycle")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed(".net"), replacement = "dotnet")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("+"), replacement = "plus")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("C#"), replacement = "csharp")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("’"), replacement = "'")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'s"), replacement = "")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("n't"), replacement = " not")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'ve"), replacement = " have")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'re"), replacement = " are")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'ll"), replacement = " will")) %>%
      mutate(Description = str_replace_all(string = Description, pattern = fixed("'d"), replacement = " would")) %>%
      unnest_tokens(output=bigram, input=Description, token = "ngrams", n = 2) %>%
      separate(bigram, c("word1", "word2"), sep = " ", remove = FALSE) %>%
      filter(!word1 %in% stop_words$word & !word2 %in% stop_words$word) %>%
      mutate(word1 = ifelse(word1 != "data", lemmatize_words(word1), word1)) %>%
      mutate(word2 = ifelse(word2 != "data", lemmatize_words(word2), word2)) %>%
      mutate(word1 = str_replace(string = word1, pattern = "analytics", replacement = "analytic")) %>%
      mutate(word2 = str_replace(string = word2, pattern = "analytics", replacement = "analytic")) %>%
      mutate(bigram = paste(word1, word2, sep=" ")) %>%
      semi_join(final_keybigrams_skill) %>%
      count(bigram) %>%
      pivot_wider(names_from = bigram, values_from = n,
                  names_prefix = "skill_")
    
    tidy_jobdesc_process <- as.data.frame(cbind.fill(tidy_jobdesc_word_exp, tidy_jobdesc_word_qual, tidy_jobdesc_word_skill, tidy_jobdesc_bigram_env, tidy_jobdesc_bigram_exp, tidy_jobdesc_bigram_qual, tidy_jobdesc_bigram_skill))
    
    tidy_jobdesc_process <- tidy_jobdesc_process[,!(names(tidy_jobdesc_process) %in% "n")]
    
    missing_cols <- data.frame(varname = as.character(names(test[, 2:ncol(test)]))) %>% anti_join(data.frame(varname = names(tidy_jobdesc_process)))
    
    for (r in 1:nrow(missing_cols)){
      columnname <- paste(missing_cols[r,1])
      add_col <- data.frame(x = 0)
      names(add_col) <- columnname
      tidy_jobdesc_process <- cbind(tidy_jobdesc_process, add_col) 
    }
    
    tidy_jobdesc_process <- tidy_jobdesc_process[,!(names(tidy_jobdesc_process) %in% "n")]
    
    multinom_result <- predict(multinom.fit, newdata=tidy_jobdesc_process)
    
    
    
    output$pred_result <- renderText({ 
      paste("This Job Description looks more like an opening for a.....",multinom_result, "!")
    })
    
    output$mytable = DT::renderDataTable({
      shownData <- as.data.frame(tidy_jobdesc_process)
      shownData
    })
  })
  
  observeEvent(input$k, {
    kcentroids = input$k
    kmeans_indeed <- kmeans(x=d[,2:ncol(d)], centers=kcentroids, nstart=25, iter.max=100)
    
    output$plot1 <- renderPlot({
      par(mar = c(5.1, 4.1, 0, 1))
      p1 <- plot(kmeans_indeed, data=d[,2:ncol(d)])
      p1
    })
  })
  
  output$plot2 <- renderPlot({
    load("wc_keys.RData")
    library(wordcloud)
    wc_keys %>%
      filter(Job_Group == "Business Analyst") %>%                            # count frequency of words
      with(wordcloud(word, n, max.words = 100, main = "Business Analyst Jobs"))  # create a word cloud w/ <= 100 words
  })
  
  output$plot3 <- renderPlot({
    load("wc_keys.RData")
    library(wordcloud)
    wc_keys %>%
      filter(Job_Group == "Cloud Engineer") %>%                            # count frequency of words
      with(wordcloud(word, n, max.words = 100, main = "Cloud Engineer Jobs"))  # create a word cloud w/ <= 100 words
  })
  
  output$plot4 <- renderPlot({
    load("wc_keys.RData")
    library(wordcloud)
    wc_keys %>%
      filter(Job_Group == "Data Analyst") %>%                            # count frequency of words
      with(wordcloud(word, n, max.words = 100, main = "Data Analyst"))  # create a word cloud w/ <= 100 words
  })
  
  output$plot5 <- renderPlot({
    load("wc_keys.RData")
    library(wordcloud)
    wc_keys %>%
      filter(Job_Group == "Network Engineer") %>%                            # count frequency of words
      with(wordcloud(word, n, max.words = 100, main = "Network Engineer"))  # create a word cloud w/ <= 100 words
  })
  
  output$plot6 <- renderPlot({
    load("wc_keys.RData")
    library(wordcloud)
    wc_keys %>%
      filter(Job_Group == "Software Developer") %>%                            # count frequency of words
      with(wordcloud(word, n, max.words = 100, main = "Software Developer"))  # create a word cloud w/ <= 100 words
  })
  
  output$plot7 <- renderPlot({
    load("wc_keys.RData")
    library(wordcloud)
    wc_keys %>%
      filter(Job_Group == "System Engineer") %>%                            # count frequency of words
      with(wordcloud(word, n, max.words = 100, main = "System Engineer"))  # create a word cloud w/ <= 100 words
  })
}

shinyApp(ui = ui, server = server)
