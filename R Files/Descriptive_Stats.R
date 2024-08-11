#Read the csv file

df <- datasets::airquality
View(df)

head(df)   #First 6 rows
head(df,10)
tail(df,2)

###########Summary of the data#########

summary(df)
df$Ozone
df$Temp
Ozone
summary(df$Temp)
attach(df)
Ozone
Temp
Month

###########DATA VISUALIZATION IN R##########
#Scatter plot

plot(x = Wind,y = Temp)
plot(Wind,Temp)
plot(Ozone, Solar.R, xlab = 'Ozone Level',
     ylab = 'Solar Values', main = 'Correlation Graph',
     col = 'red',pch = 20)


#Plot entire df

plot(df)

#Horizontal bar plot
barplot(Month)
table(Month)
barplot(table(Month))


#Histogram
hist(Ozone)
summary(df)
hist(Temp,col = '#70119c')

##Single box plot and stats

boxplot(Ozone, col = '#36e809')
boxplot(Ozone)$stats
boxplot(Ozone)$out

boxplot(Temp)

# Multiple box plots
boxplot(df)

#Multiple graphs in one canvas

par(mfrow = c(2,2), mar = c(2,2,2,2))

plot(Ozone,Temp,main = 'Scatterplot', pch = 19, col = 'red')
hist(Ozone, col = 'green')
boxplot(Ozone)
plot(Ozone, Solar.R, col = 'cyan')


#sd
sd(Temp)
mean(Temp)
#var
var(Temp)
median(Temp)
